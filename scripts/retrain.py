from src.core import train,drift
import os
if __name__=='__main__':
 result=drift()
 if result['drift_detected'] and os.getenv('AUTO_RETRAIN','false').lower()=='true':
  print('Drift detected; retraining on labeled training.csv:',train(source='scheduled_drift_trigger'))
 else:print('No retraining:',result)
