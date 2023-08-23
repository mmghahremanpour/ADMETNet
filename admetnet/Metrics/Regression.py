
from deepmodeller.Utils.Arrays  import *
from deepmodeller.Utils.Imports import *

def bootstrap_sample(ytrue, ypred, nstep, n_sample):
 
   if not arraylike(ytrue):
      ytrue = make_arraylike(ytrue)

   if not arraylike(ypred):
      ypred = make_arraylike(ypred)

   if check_array_consistency(ytrue, ypred):

      if n_sample is None or n_sample > ytrue.shape[0]:
         n_sample = ytrue.shape[0]

      yt_boot = np.zeros((nstep, n_sample))
      yp_boot = np.zeros((nstep, n_sample))

      for i in range (nstep):
         indices = np.random.choice(ytrue.shape[0], n_sample, replace=True)

         for j in range(n_sample):
            yt_boot[i][j] = ytrue[indices][j]
            yp_boot[i][j] = ypred[indices][j]

      return yt_boot, yp_boot

   else:
      return None, None

def sum_of_squared_error(ytrue, ypred):
   if not arraylike(ytrue):
      ytrue = make_arraylike(ytrue)

   if not arraylike(ypred):
      ypred = make_arraylike(ypred)

   if check_array_consistency(ytrue, ypred):
      SSE = np.sum((ytrue - ypred)**2, axis=0, dtype=np.float64)

      return SSE

   return None
   
def sum_of_absolute_error(ytrue, ypred):
   if not arraylike(ytrue):
      ytrue = make_arraylike(ytrue)

   if not arraylike(ypred):
      ypred = make_arraylike(ypred)

   if check_array_consistency(ytrue, ypred):
      SSE = np.sum(np.abs(ytrue - ypred), axis=0, dtype=np.float64)

      return SSE

   return None

def total_sum_of_squares(ytrue):
   if not arraylike(ytrue):
      ytrue = make_arraylike(ytrue)

   ytrue_mean = np.mean(ytrue, axis=0)
   SST = np.sum((ytrue - ytrue_mean)**2, axis=0, dtype=np.float64)

   return SST

def coeff_of_determination(ytrue, ypred, bootstrap=False, nstep=100, n_sample=None):
   
   if not bootstrap:
      if not arraylike(ytrue):
         ytrue = make_arraylike(ytrue)

      if not arraylike(ypred):
         ypred = make_arraylike(ypred)

      if check_array_consistency(ytrue, ypred):

         SSE = sum_of_squared_error(ytrue, ypred)
         SST = total_sum_of_squares(ytrue)

         if SST == 0:
            SST = 1e-32

         r2 = 1 - (SSE/SST)

         return r2, None

   else:

      yt_b, yp_b = bootstrap_sample(ytrue, ypred, nstep, n_sample)

      r2 = np.zeros(nstep)

      for i in range(nstep):
         yt = yt_b[i]
         yp = yp_b[i]

         yt_mean = np.average(yt, axis=0)

         SSE = sum_of_squared_error(yt, yp)
         SST = total_sum_of_squares(yt)
      
         if SST == 0:
            SST = 1e-32

         r2[i] = 1-(SSE/SST)
         

      return np.mean(r2), np.std(r2)
      

def mean_squared_error(ytrue, 
                       ypred,
                       squared=True,
                       bootstrap=False, 
                       nstep=100, 
                       n_sample=None):

   
   if not bootstrap:
      if not arraylike(ytrue):
         ytrue = make_arraylike(ytrue)

      if not arraylike(ypred):
         ypred = make_arraylike(ypred)

      if check_array_consistency(ytrue, ypred):

         SE = sum_of_squared_error(ytrue, ypred)

         MSE = SE/sample_size(ytrue)

         if not squared:
            return np.sqrt(MSE)
         else:
            return MSE, None

   else:

      yt_b, yp_b = bootstrap_sample(ytrue, ypred, nstep, n_sample)

      MSE = np.zeros(nstep)

      for i in range(nstep):
         yt = yt_b[i]
         yp = yp_b[i]

         SE = sum_of_squared_error(yt, yp)

         MSE[i] = SE/sample_size(yt)
      
      MSE_mean = np.mean(MSE) #bootstrap mean of MSE
      MSE_std  = np.std(MSE)

      if not squared:
         return np.sqrt(MSE_mean), np.sqrt(MSE_std)
      else:
         return MSE_mean, MSE_std
            

def mean_absolute_error(ytrue, 
                        ypred,
                        bootstrap=False, 
                        nstep=100, 
                        n_sample=None):

   
   if not bootstrap:
      if not arraylike(ytrue):
         ytrue = make_arraylike(ytrue)

      if not arraylike(ypred):
         ypred = make_arraylike(ypred)

      if check_array_consistency(ytrue, ypred):

         AE = sum_of_absolute_error(ytrue, ypred)

         MAE = AE/sample_size(ytrue)

         return MAE, None

   else:

      yt_b, yp_b = bootstrap_sample(ytrue, ypred, nstep, n_sample)

      MAE = np.zeros(nstep)

      for i in range(nstep):
         yt = yt_b[i]
         yp = yp_b[i]

         AE = sum_of_absolute_error(yt, yp)

         MAE[i] = AE/sample_size(yt)
      
      MAE_mean = np.mean(MAE) #bootstrap mean of MSE
      MAE_std  = np.std(MAE)

      return MAE_mean, MAE_std


def report_scores(ytrue,
                  ypred,
                  bootstrap=True,
                  nstep=100):


   @dataclass
   class stats:
      mean:float=None
      std:float=None

   r2, r2_std = coeff_of_determination(ytrue, 
                                       ypred, 
                                       bootstrap=bootstrap, 
                                       nstep=nstep)

   rmse, rmse_std = mean_squared_error(ytrue,
                                       ypred,
                                       squared=False,
                                       bootstrap=bootstrap,
                                       nstep=nstep)

   mae, mae_std = mean_absolute_error(ytrue,
                                      ypred,
                                      bootstrap=bootstrap,
                                      nstep=nstep)

   scores = {"R2": stats(*[r2, r2_std]),
             "RMSE": stats(*[rmse, rmse_std]),
             "MAE": stats(*[mae, mae_std])}

   return scores
