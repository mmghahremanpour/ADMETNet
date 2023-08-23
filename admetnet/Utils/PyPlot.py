

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
from admetnet.Utils.Imports import *

def box_plot(x, y, filename):
   plt.boxplot(x, labels=y, showmeans=True)
   plt.savefig(("%s.pdf" % filename), dpi=300)
   plt.close()

def histogram_plot(train, test, all_data, propname=None, filename=None):
   n, bins, patches = plt.hist([all_data, train, test], 50, label=['All', 'train', 'test'], color=['g', 'b', 'r'], alpha=0.5)
   plt.xlabel(propname)
   plt.legend(loc='upper center')
   plt.ylabel('Frequency')
   plt.title('Histogram of %s' % propname)
   plt.savefig(("%s.pdf" % filename), dpi=300)
   plt.close()

def scatter_plot(y, y_pred,   
                 propname="pCC50", 
                 label="Test Set", 
                 filename="scatter_plot"):

   plt.scatter(y, y_pred, c="g", alpha=0.7, marker='o')
   plt.xlabel("y")
   plt.ylabel('y_pred')
   title = "%s (%s)" % (propname, label)
   plt.title(title)
   plt.savefig(("%s.pdf" % filename), dpi=300)
   plt.close()

def scatter_plot_with_correlation_line(y, y_pred, filename="scatter_plot_correlation"):
   plt.scatter(y, y_pred, c="g", alpha=0.7, marker='o')
   axes = plt.gca()
   m, b = np.polyfit(y, y_pred, 1)
   x_corr = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
   y_corr = m*x_corr + b
   plt.plot(x_corr, y_corr, '-')
   plt.xlabel("y")
   plt.ylabel('y_pred')
   plt.savefig(("%s.pdf" % filename), dpi=300)
   plt.close()

def regression_plot(y, y_pred, 
                    propname= "pCC50", 
                    label="Test Set", 
                    filename="regression_plot"):
   """
      This function does linear regression analysis.

      y = Refrence data, e.g. experiment
      y_pred = Estimates of data obtained from a model, e.g. Neural Networks
      
   """


   """
      Fit a linear line to data. It returns the slope and the y-intercept 
      of the line. 
   """
   slope, intercept = np.polyfit(y, y_pred, 1)
   line_equation = "y={0:.2f}x+{1:.2f}".format(slope[0], intercept[0])

   ax = seaborn.regplot(x=y, y=y_pred, marker="+", ci=95, truncate=False)
   ax.set(xlabel="Experiment", ylabel = "Prediction")

   """
      Calculate the location of the line equation inside the plot
   """
   tl = ((ax.get_xlim()[1] - ax.get_xlim()[0])*0.010 + ax.get_xlim()[0],
         (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95 + ax.get_ylim()[0])
   ax.text(tl[0], tl[1], line_equation)

   title = "%s (%s)" % (propname, label)
   ax.set_title(title)
   #ax.legend()
   ax.figure.savefig(("%s.pdf" % filename), dpi=300)
   plt.close()

