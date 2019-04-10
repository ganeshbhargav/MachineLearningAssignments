1. Install anaconda for python3
	Windows: https://www.datacamp.com/community/tutorials/installing-anaconda-windows
	MacOs  : https://www.datacamp.com/community/tutorials/installing-anaconda-mac-os-x

2. Open your terminal 
   Navigate to the directory where the code is stored.
   Command: cd ~/ML/Homework1
   Run the code
   python DecisionTree.py <command line parameters>
   command line parameters are mentioned in the next step


3. As specified in assigment to run the code through command line:
	python3 <python file with decision tree code(DecisionTree.py)> <L> <K> <training_set> <validation_set> <<test_set> <to_print>


	L : Integer Value to be used in pruning
	K: Integer Value to be used in Pruning
	<training_set>: path to the training set
	<validation_set>: path to validation set
	<test_set>: path to test set
	to_print: (yes;no) to print the decision tree

	Example : python3 DecisionTree.py 3 2 data_sets1/training_set.csv data_sets1/validation_set.csv data_sets1/test_set.csv yes
