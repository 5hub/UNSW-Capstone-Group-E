Data was downloaded to loca because of size being so large

Forecast_File_Reduction.Rmd run to reduce the size of this data in order for it to fit on GitHub




Temperature_nsw.csv.zip and total_demand_nsw.csv.zip are used as inputs in Data_cleaning_mkII.Rmd.
This file verifies that our demand data is int he format that we want it to be and to merge temperature data onto it and fix any missing values.
Output is Cleaned_Data_mkII.csv -- Check that our ML and NN models use this as an input.





Cleaned_Data_mkII.csv is used as an input in Data_Exploration_mkIII.rmd to conduct data exploration and highlight potential features of the models. 
Outputs are the graphs in Exploratory Data Analysis in the project.




Cleaned_Data_mkII.csv is used in ####ML python file#### to conduct ML learning specific features and training
Outputs will be a csv file with 90% CI data for the target_test dataset.