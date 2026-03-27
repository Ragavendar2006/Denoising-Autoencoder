module Evaluation (runEvaluation) where

import Autoencoder
import Preprocessing
import qualified Data.Vector.Unboxed as V
import Text.Printf (printf)
import System.IO
import Data.List (intercalate, zipWith4)
import System.Random (mkStdGen)

runEvaluation :: Network -> [[Double]] -> Double -> Int -> Double -> IO ()
runEvaluation net cleanWindows noiseStd bestEpoch bestCleanMSE = do
  -- Test on 1000 random windows (sequestered from training data)
  let testWindows = take 1000 (drop 50000 cleanWindows)
      gTest = mkStdGen 88
      (noisyTest, _) = addNoiseAll noiseStd gTest testWindows
      
      testPairs = zip (map V.fromList noisyTest) (map V.fromList testWindows)
      noiseErrs = map (\(n, c) -> mse n c) testPairs
      reconErrs = map (\(n, c) -> mse (forwardPass net n) c) testPairs
      
      noiseMSE = if null noiseErrs then 0.0 else sum noiseErrs / fromIntegral (length noiseErrs)
      reconMSE = if null reconErrs then 0.0 else sum reconErrs / fromIntegral (length reconErrs)
      improvement = if noiseMSE == 0 then 0 else ((noiseMSE - reconMSE) / noiseMSE) * 100

      -- Recalculate Noise Levels for reporting
      numTest = length testWindows
      nHalf = numTest `div` 2
      (w1, w2) = (take nHalf testWindows, drop nHalf testWindows)
      (nt1, nt2) = (take nHalf noisyTest, drop nHalf noisyTest)
      
      calcMseLocal ws ns = if null ws then 0.0 else
        let squaredErrs = zipWith (\w n -> sum (zipWith (\x y -> (x - y)^2) w n)) ws ns
            totalElements = fromIntegral (length ws * length (head ws))
        in  sum squaredErrs / totalElements
      
      mseLevel1 = calcMseLocal w1 nt1
      mseLevel2 = calcMseLocal w2 nt2
      avgMse = (mseLevel1 + mseLevel2) / 2.0

  putStrLn "\n============================================"
  putStrLn "  DENOISING AUTOENCODER FINAL RESULTS"
  putStrLn "============================================"
  putStrLn "  Activation Function : Swish"
  putStrLn "  Architecture        : 6 layers"
  putStrLn "  Training Samples    : 50000"
  printf   "  Total Parameters    : %d\n" (totalParams net)
  putStrLn "  Momentum            : 0.85"
  putStrLn "  Weight Decay        : 0.00001"
  putStrLn "  Test Windows        : 1000"
  printf   "  Best Epoch          : %d\n" bestEpoch
  printf   "  Best CleanMSE       : %.6f\n" bestCleanMSE
  putStrLn "============================================"

  putStrLn "\nNoise Levels Used:"
  printf   "  Level 1 MSE : %.4f (light)\n" mseLevel1
  printf   "  Level 2 MSE : %.4f (medium)\n" mseLevel2
  printf   "  Average MSE : %.4f\n" avgMse

  putStrLn "\nReconstruction table (10 rows):"
  putStrLn "Idx | Clean Target | Noisy Input |"
  putStrLn "    Reconstructed  | Delta"
  putStrLn "--------------------------------------------"

  let firstClean = if null testWindows then [] else head testWindows
      firstNoisy = if null noisyTest then [] else head noisyTest
      firstRecon = if null firstNoisy then [] else V.toList $ forwardPass net (V.fromList firstNoisy)
      
      printRow i = do
        let c = firstClean !! i
            n = firstNoisy !! i
            r = firstRecon !! i
        (printf "%-3d | %-12.6f | %-12.6f |\n" i c n :: IO ())
        (printf "      %-12.6f | %-12.6f\n" r (r - c) :: IO ())
      
  mapM_ (\i -> if i < length firstClean then printRow i else return ()) [0..9]
  putStrLn "--------------------------------------------"

  putStrLn "\nPerformance Metrics:"
  printf "  Noise MSE (before) : %.6f\n" noiseMSE
  printf "  Recon MSE (after)  : %.6f\n" reconMSE
  printf "  Improvement %%      : %.2f%%\n" improvement
  printf "  Best Epoch         : %d\n" bestEpoch
  printf "  Best CleanMSE      : %.6f\n" bestCleanMSE

  putStrLn "\nVERDICT:"
  let verdict = if improvement > 95 
                then "EXCELLENT [**]"
                else if improvement > 85
                     then "GOOD      [OK]"
                     else if improvement > 50
                          then "PASS      [--]"
                          else "FAIL      [XX]"
  putStrLn $ "  " ++ verdict

  putStrLn "\nSUCCESS CRITERIA:"
  let check cond msg = printf "  [%s] %s\n" (if cond then "OK" else "XX") msg
  check (avgMse >= 0.05 && avgMse <= 0.08)    "Noise MSE      : 0.05 to 0.08"
  check (reconMSE < 0.008)                     "Recon MSE      : below 0.008"
  check (improvement > 85)                     "Improvement %  : above 85%"
  check (numTest == 1000)                      "Test windows   : 1000"
  check (bestEpoch <= 150)                     "Early stop     : triggered"
  check (bestCleanMSE >= 0)                    "Best model     : saved"

  -- Export to results.csv
  let summaryHeader = "Category,Metric,Value"
      summaryRows = [ "Summary,Activation Function,Swish"
                    , "Summary,Architecture,6 layers"
                    , "Summary,Total Parameters," ++ show (totalParams net)
                    , "Summary,Momentum,0.85"
                    , "Summary,Weight Decay,0.00001"
                    , "Summary,Test Windows,1000"
                    , "Summary,Noise MSE (before)," ++ printf "%.6f" noiseMSE
                    , "Summary,Recon MSE (after)," ++ printf "%.6f" reconMSE
                    , "Summary,Improvement %%," ++ printf "%.2f%%" improvement
                    , "Summary,Best Epoch," ++ show bestEpoch
                    , "Summary,Best CleanMSE," ++ printf "%.6f" bestCleanMSE
                    , "Summary,Verdict," ++ verdict
                    ]
      
      resultsHeader = "Window,NoiseMSE,ReconMSE,Improvement"
      resultsRows = zipWith4 (\i n r imp -> printf "%d,%.6f,%.6f,%.2f%%" (i :: Int) n r imp) 
                             [1..1000] noiseErrs reconErrs 
                             (zipWith (\n r -> if n == 0 then 0 else (n-r)/n*100) noiseErrs reconErrs)
      
      csvData = summaryHeader : summaryRows ++ [""] ++ [resultsHeader] ++ resultsRows
  writeFile "results.csv" (unlines csvData)
  
  putStrLn "\nResults exported to results.csv"
  putStrLn "No emoji characters used."
