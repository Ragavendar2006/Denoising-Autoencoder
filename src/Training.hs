{-# LANGUAGE BangPatterns #-}

module Training (sanityCheck, fullTraining) where

import Autoencoder
import Preprocessing
import System.Random
import qualified Data.Vector.Unboxed as V
import Text.Printf (printf)
import System.Exit (exitFailure)
import Data.List (foldl', minimumBy)
import Data.Ord (comparing)
import Control.Monad (when)
import Control.Parallel.Strategies (parMap, rdeepseq)

-- | Momentum and Weight Decay Update
applyMomentumAndDecay :: Double -> Network -> Gradients -> Gradients -> (Network, Gradients)
applyMomentumAndDecay lr (Network layers) (Gradients grads) (Gradients vels) =
  let results = zip3 layers grads vels
      (!layers', !vels') = unzip $ map updateLayer results
  in  (Network layers', Gradients vels')
  where
    updateLayer (Layer ws bs inSz outSz, LayerGrad gW gB, LayerGrad vW vB) =
      let -- Momentum (0.85)
          vW' = V.zipWith (\v g -> 0.85 * v - lr * g) vW gW
          vB' = V.zipWith (\v g -> 0.85 * v - lr * g) vB gB
          -- Weights with Decay (0.00001)
          ws' = V.zipWith (\w v -> w * (1.0 - 0.00001) + v) ws vW'
          -- Biases (No decay)
          bs' = V.zipWith (+) bs vB'
      in (Layer ws' bs' inSz outSz, LayerGrad vW' vB')

-- | Sanity check: Overfit on a single window to 0.0 MSE
sanityCheck :: StdGen -> [Double] -> IO Bool
sanityCheck g1 cleanWindow = do
  putStrLn "\n===== SANITY CHECK ====="
  let noisy = V.fromList cleanWindow
      clean = V.fromList cleanWindow
      archSpec = [(64, 128), (128, 256), (256, 128), (128, 64), (64, 128), (128, 64)]
  (net0, _) <- initNetwork g1 archSpec
  let lr = 1e-2
      iters = 300
      vels0 = zeroGrads net0

  -- Run iterations
  finalNet <- trainLoop net0 vels0 noisy clean lr iters 1
  
  let output = forwardPass finalNet noisy
      finalMSE = mse output clean
      
  putStrLn ""
  if finalMSE < 0.001
    then do
      (printf "[SANITY CHECK PASSED] MSE: %.6f\n" finalMSE :: IO ())
      return True
    else do
      (printf "[SANITY CHECK FAILED] MSE: %.6f\n" finalMSE :: IO ())
      return False

trainLoop :: Network -> Gradients -> V.Vector Double -> V.Vector Double -> Double -> Int -> Int -> IO Network
trainLoop !net !vels _ _ _ 0 _ = return net
trainLoop !net !vels !noisy !clean !lr !remaining !iterCount = do
  let grads = backward net noisy clean
      (!net', !vels') = applyMomentumAndDecay lr net grads vels
      output = forwardPass net' noisy
      currentMSE = mse output clean
         
  if iterCount `mod` 20 == 0
    then (printf "Iter %-3d MSE: %.6f\n" iterCount currentMSE :: IO ())
    else return ()
    
  trainLoop net' vels' noisy clean lr (remaining - 1) (iterCount + 1)

data TrainState = TrainState
  { tsNet           :: !Network
  , tsVelocity      :: !Gradients
  , tsBestNet       :: !Network
  , tsBestMSE       :: !Double
  , tsPatience      :: !Int
  , tsMaxPatience   :: !Int
  , tsLR            :: !Double
  , tsBestEpoch     :: !Int
  }

fullTraining :: StdGen -> [[Double]] -> Double -> Int -> Double -> IO (Int, Double)
fullTraining g cleanWindows lr0 batchSize noiseStd = do
  putStrLn "\n===== FULL TRAINING STAGE ====="
  
  let archSpec = [(64, 128), (128, 256), (256, 128), (128, 64), (64, 128), (128, 64)]
  (net0, g1) <- initNetwork g archSpec
  
  let trainWindows = take 50000 cleanWindows
      (noisyData, _) = addNoiseAll noiseStd g1 trainWindows
      pairs = zip (map V.fromList noisyData) (map V.fromList trainWindows)
      batches = chunkList batchSize pairs
      
      initialState = TrainState 
        { tsNet = net0
        , tsVelocity = zeroGrads net0
        , tsBestNet = net0
        , tsBestMSE = 1e10
        , tsPatience = 0
        , tsMaxPatience = 20
        , tsLR = lr0
        , tsBestEpoch = 0
        }

  runLoop 1 initialState batches pairs

  where
    runLoop epoch state batches pairs
      | epoch > 150 = finish state
      | otherwise = do
          -- LR Decay: every 20 epochs multiply LR by 0.7
          let (lr, msgLR) = if epoch > 1 && (epoch - 1) `mod` 20 == 0
                            then (tsLR state * 0.7, printf "[LR DECAY] Epoch %d New LR: %.2e\n" epoch (tsLR state * 0.7) :: IO ())
                            else (tsLR state, return ())
          msgLR
          
          -- Train one epoch
          let (!net', !vels', !lastGradNorm) = trainEpochWithNormMom (tsNet state) (tsVelocity state) lr batches
          
          -- Evaluate
          let !currentMSE = evalMSE net' pairs
          
          -- Early Stopping logic + SAVE
          let !improvement = tsBestMSE state - currentMSE
              !threshold = 0.00001
          
          (!bestNet', !bestMSE', !patience', !bestEpoch') <- 
            if improvement > threshold
            then do
              printf "[SAVED] New best at epoch %d \n         CleanMSE: %.6f\n" epoch currentMSE
              saveModel "model.dat" net'
              return (net', currentMSE, 0, epoch)
            else return (tsBestNet state, tsBestMSE state, tsPatience state + 1, tsBestEpoch state)

          -- Print Status
          printf "Epoch %d/150  CleanMSE: %.6f\n" epoch currentMSE
          printf "Patience: %d/20  LR: %.2e\n" patience' (tsLR state)
          printf "GradNorm: %.6f\n" lastGradNorm
          putStrLn "Momentum: active"
          printf "Best so far: %.6f\n" bestMSE'
          when (lastGradNorm > 3.0) $ putStrLn "WARNING: Momentum instability detected"

          -- Stop or Continue
          if patience' >= tsMaxPatience state
          then do
            printf "[EARLY STOP] Stopping at epoch %d\n" epoch
            finish (state { tsBestNet = bestNet', tsBestMSE = bestMSE', tsBestEpoch = bestEpoch' })
          else runLoop (epoch + 1) (state { tsNet = net', tsVelocity = vels', tsBestNet = bestNet', tsBestMSE = bestMSE', tsPatience = patience', tsLR = lr, tsBestEpoch = bestEpoch' }) batches pairs

    finish state = do
      putStrLn ""
      saveModel "model.dat" (tsBestNet state)
      printf "Best epoch    : %d\n" (tsBestEpoch state)
      printf "Best CleanMSE : %.6f\n" (tsBestMSE state)
      return (tsBestEpoch state, tsBestMSE state)

trainEpochWithNormMom :: Network -> Gradients -> Double -> [[(V.Vector Double, V.Vector Double)]] -> (Network, Gradients, Double)
trainEpochWithNormMom !net !vels !lr !batches = go net vels 0.0 batches
  where
    go !n !v !lastG [] = (n, v, lastG)
    go !n !v _ (b:bs) = 
      let !sz = fromIntegral (length b)
          -- Parallelize backward pass
          !gradsList = parMap rdeepseq (\(!nx, !cx) -> backward n nx cx) b
          !batchGrad = scaleGrads (1.0 / sz) (foldl' addGrads (zeroGrads n) gradsList)
          !gNorm = gradNorm batchGrad
          (!n', !v') = applyMomentumAndDecay lr n batchGrad v
      in  go n' v' gNorm bs

evalMSE :: Network -> [(V.Vector Double, V.Vector Double)] -> Double
evalMSE net pairs = 
  let !errs = map (\(!n, !c) -> mse (forwardPass net n) c) pairs
  in  sum errs / fromIntegral (length pairs)

chunkList :: Int -> [a] -> [[a]]
chunkList _ [] = []
chunkList n xs = take n xs : chunkList n (drop n xs)
