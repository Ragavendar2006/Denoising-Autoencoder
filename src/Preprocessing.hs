module Preprocessing
  ( loadData
  , preprocessData
  , slidingWindows
  , normalizeWindow
  , addNoise
  , addNoiseAll
  ) where

import System.Random
import qualified Data.Vector.Unboxed as V
import Text.Printf (printf)
import Debug.Trace (trace)

loadData :: FilePath -> IO [Double]
loadData path = do
  content <- readFile path
  return [read x | x <- lines content, not (null x)]

slidingWindows :: Int -> [a] -> [[a]]
slidingWindows n xs =
  let window = take n xs
  in if length window == n
     then window : slidingWindows n (drop 1 xs)
     else []

normalizeWindow :: [Double] -> [Double]
normalizeWindow window =
  let minVal = minimum window
      maxVal = maximum window
      rng = maxVal - minVal
  in  if rng < 1e-10
      then replicate (length window) 0.0
      else map (\x -> (x - minVal) / rng - 0.5) window

preprocessData :: [Double] -> [[Double]]
preprocessData rawData =
  let windows = slidingWindows 64 rawData
      normalized = map normalizeWindow windows
  in  take 100000 normalized

-- Note: We use the std argument from callers (like Main/Training)
-- If we want to hit 0.05-0.08 specifically via Preprocessing.hs,
-- we might need to override the input 'std' if it's too low.
addNoise :: Double -> StdGen -> [Double] -> ([Double], StdGen)
addNoise std g window = addNoiseInternal std 0.15 0.5 g window

addNoiseInternal :: Double -> Double -> Double -> StdGen -> [Double] -> ([Double], StdGen)
addNoiseInternal std spikeProb spikeMag g window =
  let (gaussianNoise, g1) = gaussianVec g (length window) std
      (spiked, g2) = addSpikes g1
      noisyList = zipWith3 (\c gn sn -> c + gn + sn) window gaussianNoise spiked
  in  (noisyList, g2)
  where
    addSpikes :: StdGen -> ([Double], StdGen)
    addSpikes gen = go gen (length window) []
      where
        go gen' 0 acc = (reverse acc, gen')
        go gen' rem acc =
          let (p, gen1) = randomR (0.0 :: Double, 1.0) gen'
              (signP, gen2) = randomR (0 :: Int, 1) gen1
              spike = if p < spikeProb then (if signP == 0 then -spikeMag else spikeMag) else 0.0
          in  go gen2 (rem - 1) (spike : acc)

gaussianVec :: StdGen -> Int -> Double -> ([Double], StdGen)
gaussianVec g n std = go g n []
  where
    go gen 0 acc = (reverse acc, gen)
    go gen rem acc =
      let (u1, gen1) = randomR (1e-10, 1.0) gen
          (u2, gen2) = randomR (0.0, 1.0) gen1
          z0 = std * sqrt (-2.0 * log u1) * cos (2.0 * pi * u2)
          z1 = std * sqrt (-2.0 * log u1) * sin (2.0 * pi * u2)
      in  if rem >= 2
          then go gen2 (rem - 2) (z1 : z0 : acc)
          else go gen2 (rem - 1) (z0 : acc)

addNoiseAll :: Double -> StdGen -> [[Double]] -> ([[Double]], StdGen)
addNoiseAll _ g windows =
  let (noisyTotal, gFinal, mse1, mse2, avgMSE) = addNoiseCalibrated 0.15 0.25 g windows 0
      
      logMsg = printf "Noise Level 1 MSE : %.4f (light)\n" mse1
            ++ printf "Noise Level 2 MSE : %.4f (medium)\n" mse2
            ++ printf "Average Noise MSE : %.4f" avgMSE
  in trace logMsg (noisyTotal, gFinal)
  where
    addNoiseCalibrated s1 s2 gn ws depth =
      let n = length ws
          n1 = n `div` 2
          (w1, w2) = (take n1 ws, drop n1 ws)
          
          -- Level 1: Light (0.15 base std, 8% prob, 0.35 mag)
          (noisy1, gn1) = applyNoiseLevel s1 0.08 0.35 gn w1
          -- Level 2: Medium (0.25 base std, 15% prob, 0.50 mag)
          (noisy2, gn2) = applyNoiseLevel s2 0.15 0.50 gn1 w2
          
          m1 = calcNoiseMSE w1 noisy1
          m2 = calcNoiseMSE w2 noisy2
          avg = (m1 + m2) / 2.0
          
      in if depth < 3 && avg < 0.05 then
           trace "WARNING: Noise too low" $ addNoiseCalibrated (s1 + 0.05) (s2 + 0.05) gn ws (depth + 1)
         else if depth < 3 && avg > 0.10 then
           trace "WARNING: Noise too high" $ addNoiseCalibrated (max 0.01 (s1 - 0.05)) (max 0.01 (s2 - 0.05)) gn ws (depth + 1)
         else (noisy1 ++ noisy2, gn2, m1, m2, avg)

    applyNoiseLevel std prob mag gen ws = go gen ws []
      where
        go gnn [] acc = (reverse acc, gnn)
        go gnn (x:xs) acc = 
          let (nx, gnn') = addNoiseInternal std prob mag gnn x
          in  go gnn' xs (nx : acc)

    calcNoiseMSE ws ns =
      if null ws then 0.0 else
      let squaredErrs = zipWith (\w n -> sum (zipWith (\x y -> (x - y)^2) w n)) ws ns
          totalElements = fromIntegral (length ws * length (head ws))
      in  sum squaredErrs / totalElements
