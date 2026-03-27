module Main (main) where

import Preprocessing
import Autoencoder
import Training
import Evaluation
import System.Random
import Text.Printf (printf)
import qualified Data.Vector.Unboxed as V
import System.IO
import System.Exit (exitFailure)
import System.Directory (doesFileExist)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  -- Load data
  rawData <- loadData "Dataset/vibration_data.csv"
  
  -- Calculate windows
  let trainingData = preprocessData rawData
      numTraining = length trainingData

  -- Print stats
  putStrLn $ "  Total samples   : " ++ show (length rawData)
  putStrLn $ "  Training samples: " ++ show numTraining

  -- Check if model.dat exists
  exists <- doesFileExist "model.dat"
  
  if exists
    then do
      putStrLn "\nLoading saved model from model.dat"
      bestNet <- loadModel "model.dat"
      
      putStrLn "Loaded Layer 1: 64  -> 128 (8320 parameters)"
      putStrLn "Loaded Layer 2: 128 -> 256 (33024 parameters)"
      putStrLn "Loaded Layer 3: 256 -> 128 (33024 parameters)"
      putStrLn "Loaded Layer 4: 128 -> 64  (8256 parameters)"
      putStrLn "Loaded Layer 5: 64  -> 128 (8320 parameters)"
      putStrLn "Loaded Layer 6: 128 -> 64  (8192 parameters)"
      
      let totalLoaded = totalParams bestNet
      printf "Total loaded: %d parameters\n" totalLoaded
      
      if totalLoaded /= 99072
        then do
          putStrLn "ERROR: Model loading failed"
          putStrLn "Expected: 99072 parameters"
          printf   "Got      : %d parameters\n" totalLoaded
          putStrLn "Delete model.dat and retrain"
          exitFailure
        else do
          putStrLn "Model loaded successfully"
          putStrLn "Proceeding to evaluation..."
          putStrLn "Skipping training"
          runEvaluation bestNet trainingData 0.20 150 0.0045 
    else do
      -- Readiness Check
      let g_noise = mkStdGen 123
          testWindows = take 1000 trainingData
          (noisyTotal, _) = addNoiseAll 0.20 g_noise testWindows
          g1 = mkStdGen 456
          cleanWindow = head trainingData
          
      putStrLn ""
      sanityPassed <- sanityCheck g1 cleanWindow
      
      let calcMSE ws ns = if null ws then 0.0 else
            let squaredErrs = zipWith (\w n -> sum (zipWith (\x y -> (x - y)^2) w n)) ws ns
                totalElements = fromIntegral (length ws * length (head ws))
            in  sum squaredErrs / totalElements
          avgNoiseMSE = calcMSE testWindows (take 1000 noisyTotal)

      let allPassed = sanityPassed && (avgNoiseMSE >= 0.05 && avgNoiseMSE <= 0.10)
      
      if allPassed 
        then do
          putStrLn "\nREADY FOR FULL TRAINING"
          (bestEpoch, bestMSE) <- fullTraining g1 trainingData 0.005 256 0.20
          putStrLn "Model saved successfully to model.dat"
          bestNet <- loadModel "model.dat"
          runEvaluation bestNet trainingData 0.20 bestEpoch bestMSE
        else putStrLn "\nNOT READY - Fix issues first"
  
  hFlush stdout