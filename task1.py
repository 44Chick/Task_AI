from prediction_util import PredictionUtil as pu

white = pu()

white.read('winequality-white.csv')

white.run_svm2(["pH","sulphates","alcohol"],"quality")
white.run_decision_tree_classifier2(["pH","sulphates","alcohol"],"quality")

white.lmplot('alcohol','pH','quality')
white.plot_3d('alcohol','pH','quality')
white.myviolinplot("quality","alcohol")
white.heatmap1()
white.histogram()




