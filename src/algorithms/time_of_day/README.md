# Event Times Only

## Purpose

Simple algorithm using time of day likelihood to generate risk forecast.
Example: If all previous seizures were at 9am, the likelihood of a seizure at 9am would be 1 and 0 at all other times.

## Required Inputs

patient_id
seizure_events.start_time

## Hyperparameters

- **threshold_increment**: float (default: 0.01) - increment between risk score thresholds to search
  over (eg default leads to threshold options: [0.01, 0.02, 0.03...0.99])
