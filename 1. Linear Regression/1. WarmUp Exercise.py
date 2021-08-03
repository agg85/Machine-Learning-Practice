import os								#for manipulating directory paths
import numpy as np						#vectors and arrays
from matplotlib import pyplot 			#plotting lib
from mpl_toolkits.mplot3d import Axes3D #for plotting 3D surfaces

#assignment submission and grading related
import utils
grader = utils.Grader()

# %matplotlib inline

################# WarmUpExercise #########################################

def warmUpExercise():
	return np.identity(5)

warmUpExercise()

grader[1] = warmUpExercise
# grader.grade()