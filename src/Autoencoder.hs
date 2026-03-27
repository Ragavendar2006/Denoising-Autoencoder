{-# LANGUAGE BangPatterns, DeriveGeneric #-}
module Autoencoder
  ( Network(..)
  , Layer(..)
  , Gradients(..)
  , LayerGrad(..)
  , initNetwork
  , forwardPass
  , backward
  , applyGradients
  , gradNorm
  , addGrads
  , scaleGrads
  , zeroGrads
  , loadModel
  , saveModel
  , swish
  , swishDeriv
  , linear
  , linearDeriv
  , mse
  , totalParams
  ) where

import qualified Data.Vector.Unboxed as V
import System.Random (StdGen, randomR)
import Data.List (foldl')
import Text.Printf (printf)
import Data.Binary
import qualified Data.ByteString.Lazy as BL
import Control.DeepSeq (NFData(..), rnf)
import GHC.Generics (Generic)

data Layer = Layer
  { lWeights :: !(V.Vector Double)
  , lBiases  :: !(V.Vector Double)
  , lInSize  :: !Int
  , lOutSize :: !Int
  } deriving (Show, Generic)

instance NFData Layer

instance Binary Layer where
  put l = do
    put (V.toList (lWeights l))
    put (V.toList (lBiases l))
    put (lInSize l)
    put (lOutSize l)
  get = do
    ws <- get
    bs <- get
    ins <- get
    outs <- get
    return $ Layer (V.fromList ws) (V.fromList bs) ins outs

data Network = Network ![Layer] deriving (Show, Generic)

instance NFData Network

instance Binary Network where
  put (Network ls) = put ls
  get = Network <$> get

saveModel :: FilePath -> Network -> IO ()
saveModel path net = BL.writeFile path (encode net)

loadModel :: FilePath -> IO Network
loadModel path = decode <$> BL.readFile path

type Weights = Network
type Input = V.Vector Double
type Output = V.Vector Double

initNetwork :: StdGen -> [(Int, Int)] -> IO (Network, StdGen)
initNetwork g specs = do
  (layers, gFinal, totalParams) <- go g specs 1 0 []
  (printf "  Total parameters: %d\n" totalParams :: IO ())
  return (Network (reverse layers), gFinal)
  where
    go gen [] _ total acc = return (acc, gen, total)
    go gen ((fanIn, fanOut):rest) idx total acc = do
      let std = sqrt (2.0 / fromIntegral (fanIn + fanOut))
          nWeights = fanIn * fanOut
          (ws, gen') = randomVec gen nWeights std
          bs = V.replicate fanOut 0.0
          layerParams = nWeights + fanOut
      
      (printf "  Layer %d: %-3d -> %-3d initialized\n" (idx :: Int) fanIn fanOut :: IO ())
      go gen' rest (idx + 1) (total + layerParams) (Layer ws bs fanIn fanOut : acc)

-- Box-muller random generation
randomVec :: StdGen -> Int -> Double -> (V.Vector Double, StdGen)
randomVec g n std =
  let go !gen 0 !acc = (V.fromList (reverse acc), gen)
      go !gen !remSteps !acc =
        let (!u1, !gen1) = randomR (1e-10, 1.0) gen
            (!u2, !gen2) = randomR (0.0, 1.0) gen1
            !z0 = std * sqrt (-2.0 * log u1) * cos (2.0 * pi * u2)
            !z1 = std * sqrt (-2.0 * log u1) * sin (2.0 * pi * u2)
        in  if remSteps >= 2
            then go gen2 (remSteps - 2) (z1 : z0 : acc)
            else go gen2 (remSteps - 1) (z0 : acc)
  in  go g n []

swish :: Double -> Double
swish x = x * (1.0 / (1.0 + exp (-x)))

swishDeriv :: Double -> Double
swishDeriv x =
  let sig = 1.0 / (1.0 + exp (-x))
  in sig * (1.0 + x * (1.0 - sig))

linear :: Double -> Double
linear x = x

linearDeriv :: Double -> Double
linearDeriv _ = 1.0

forwardLayer :: Layer -> V.Vector Double -> V.Vector Double
forwardLayer (Layer !ws !bs !inSz !outSz) !x =
  V.generate outSz $ \i ->
    let !rowStart = i * inSz
        !dot = sumProd ws x rowStart inSz
    in  dot + (bs `V.unsafeIndex` i)

sumProd :: V.Vector Double -> V.Vector Double -> Int -> Int -> Double
sumProd !ws !xs !start !len = go 0 0.0
  where go !j !acc | j >= len = acc
                   | otherwise = go (j + 1) (acc + ws `V.unsafeIndex` (start + j) * xs `V.unsafeIndex` j)

forwardPass :: Weights -> Input -> Output
forwardPass (Network layers) !input = go 0 layers input
  where
    n = length layers
    go _ [] !x = x
    go !idx (l:ls) !x =
      let !z = forwardLayer l x
          !a = if idx < n - 1 then V.map swish z else V.map linear z
      in  go (idx + 1) ls a

forwardStore :: Network -> V.Vector Double -> ([(V.Vector Double, V.Vector Double)], V.Vector Double)
forwardStore (Network layers) !input = go 0 layers input []
  where
    n = length layers
    go _ [] !x !acc = (reverse acc, x)
    go !idx (l:ls) !x !acc =
      let !z = forwardLayer l x
          !a = if idx < n - 1 then V.map swish z else V.map linear z
      in  go (idx + 1) ls a ((z, a) : acc)

data LayerGrad = LayerGrad
  { lgWeights :: !(V.Vector Double)
  , lgBiases  :: !(V.Vector Double)
  } deriving (Generic)

instance NFData LayerGrad

data Gradients = Gradients ![LayerGrad] deriving (Generic)

instance NFData Gradients

backward :: Network -> V.Vector Double -> V.Vector Double -> Gradients
backward net@(Network layers) !input !target =
  let (!stored, !out) = forwardStore net input
      -- Assuming n is batch size, which is 1
      !nLen = 1.0
      -- Step 1: Output delta (Layer 4)
      !delta4 = V.zipWith (\o t -> (o - t) * (2.0 / nLen)) out target
      
      numLayers = length layers
      layerInputs = input : map snd (init stored)
      triples = zip3 [0..] layers (zip (map fst stored) layerInputs)
      (!grads, _) = foldr (backStep numLayers) ([], delta4) triples
  in  Gradients grads

backStep :: Int -> (Int, Layer, (V.Vector Double, V.Vector Double)) 
         -> ([LayerGrad], V.Vector Double) -> ([LayerGrad], V.Vector Double)
backStep !numLayers (!idx, !layer, (!preAct, !layerIn)) (!acc, !deltaNext) =
  let -- If idx == numLayers - 1, we are at Linear output layer. deltaNext is delta4.
      -- If idx < numLayers - 1, we are at Swish layer. We already passed deltaNext.
      !deltaCurrent = if idx < numLayers - 1 
                      then V.zipWith (*) deltaNext (V.map swishDeriv preAct) 
                      else deltaNext
      
      !inSz = lInSize layer; !outSz = lOutSize layer
      
      -- grad_W = deltaCurrent * transpose(layerIn)
      !wGrad = V.generate (outSz * inSz) $ \k ->
        (deltaCurrent `V.unsafeIndex` (k `div` inSz)) * (layerIn `V.unsafeIndex` (k `mod` inSz))
      !bGrad = deltaCurrent
      
      -- Next delta for lower layer: transpose(W) * deltaCurrent
      !deltaPrev = V.generate inSz $ \j ->
        let go !i !v | i >= outSz = v
                     | otherwise = go (i + 1) (v + lWeights layer `V.unsafeIndex` (i * inSz + j) * deltaCurrent `V.unsafeIndex` i)
        in  go 0 0.0
  in  (LayerGrad wGrad bGrad : acc, deltaPrev)

applyGradients :: Double -> Network -> Gradients -> Network
applyGradients !lr (Network layers) (Gradients grads) =
  Network $ zipWith applyLayer layers grads
  where
    applyLayer !l !g =
      let !ws' = V.zipWith (\w dw -> w - lr * clamp dw) (lWeights l) (lgWeights g)
          !bs' = V.zipWith (\b db -> b - lr * clamp db) (lBiases l) (lgBiases g)
      in  l { lWeights = ws', lBiases = bs' }
    clamp x = max (-5.0) (min 5.0 x)

gradNorm :: Gradients -> Double
gradNorm (Gradients grads) =
  let !sqSum = foldl' (\acc lg -> acc + V.sum (V.map (\x -> x*x) (lgWeights lg)) 
                                      + V.sum (V.map (\x -> x*x) (lgBiases lg))) 0.0 grads
  in  sqrt sqSum

mse :: V.Vector Double -> V.Vector Double -> Double
mse o t = V.sum (V.zipWith (\x y -> (x - y) ^ (2 :: Int)) o t) / fromIntegral (V.length o)

addGrads :: Gradients -> Gradients -> Gradients
addGrads (Gradients g1) (Gradients g2) = 
  Gradients (zipWith (\(LayerGrad w1 b1) (LayerGrad w2 b2) -> LayerGrad (V.zipWith (+) w1 w2) (V.zipWith (+) b1 b2)) g1 g2)

scaleGrads :: Double -> Gradients -> Gradients
scaleGrads s (Gradients gs) =
  Gradients (map (\(LayerGrad w b) -> LayerGrad (V.map (*s) w) (V.map (*s) b)) gs)

zeroGrads :: Network -> Gradients
zeroGrads (Network layers) =
  Gradients (map (\l -> LayerGrad (V.replicate (V.length (lWeights l)) 0.0) (V.replicate (V.length (lBiases l)) 0.0)) layers)

totalParams :: Network -> Int
totalParams (Network layers) = sum (map (\l -> V.length (lWeights l) + V.length (lBiases l)) layers)
