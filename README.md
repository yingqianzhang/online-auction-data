# online-auction-data

The codes and data were used in the paper:
- Ye, Q. C., Rhuggenaath, J., Zhang, Y., Verwer, S., & Hilgeman, M. J. (2021). Data driven design for online industrial auctions. Annals of Mathematics and Artificial Intelligence, 1-17.  


How to run:
- python auction_opt_AMAI.py

Packages needed:
- Cplex: used to solve the mixed linear programming (MIP) model for optimization
- sklearn, pandas, numpy
- 

Python files:
- The file "auction_opt_AMAI.py" optimizes the auction using a MIP.
	It reads the training data needed for training a decision tree.
	Then it formulate the design of an auction as a MIP model, where some constraints are from the learned decision tree.
	Then it uses Cplex solver to solve the MIP.
	Then it does some post processing to further optimize the design of an auction (using the file "postprocess_MIP_sol_AMAI.py").
- The file "postprocess_MIP_sol_AMAI.py" does some post-processing to further optimize the design of an auction.


Input data files, in folder "data_3":
	- LP_tree_FULL_test_data_1.csv
	- LP_tree_FULL_test_data_2.csv
	- LP_tree_FULL_test_data_3.csv
	- LP_tree_FULL_test_data_4.csv
	- LP_tree_FULL_test_data_5.csv
	- LP_tree_optimize_test_data_1.csv
	- LP_tree_optimize_test_data_2.csv
	- LP_tree_optimize_test_data_3.csv
	- LP_tree_optimize_test_data_4.csv
	- LP_tree_optimize_test_data_5.csv
	- LP_tree_train_data.csv


Output data files:
-	If you run "auction_opt_AMAI.py" with "test_nr = X" this will create the following files in the folder "data_3":
		- RESULT_DESIGN_v1_test_X.xlsx
		- RESULT_DESIGN_v1_TBA_test_X.xlsx
-	The file "RESULT_DESIGN_v1_TBA_test_X.xlsx" contains the result of the auction optimization. 
	The column "LotNr_NEW_sorted" contains the new LotNr.
	The column "SP_NEW" contains the new value for the starting price (SP).
	The column "yClass_MIP" contains the class prediction after optimization with  the MIP and after post-processing.
	The column "yClass_pred" contains the class prediction before optimization with  the MIP.
