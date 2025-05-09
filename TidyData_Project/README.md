# Tidy Data Project
For references and instructions, scroll to the bottom.

**Tidy Data principles are all about creating a standard and clean way to present data**
I chose the 2008 Olympic Medalist 🎖️ data for no particular reason and began to clean.
The first thing I do in this project is to tidy up the data - first by separating the names from the event categories, then by melting the data set, and finally by dropping those events in which an athlete did not medal (eliminating fruitless observations). Then, I created a pivot plot to look at the data from a different perspective and created _three_ different visualizations (from the un-pivoted data frame).

**My Goal was simply to explore the data and to see what I could learn about the data set by manipulating data frames**
I had hoped to be able to determine who had won the most medals, and specifically the most gold medals, but was unable to do so because the medaling categories were already placed into arbitrary bins in the dataset.

Here is an example image that shows the number of medal categories won by any person:
![image](https://github.com/user-attachments/assets/3a6c0ba2-2404-4c64-ab07-8cde2e46079c)

As you can see, every medalist only won one of these arbitrary categories. 

The pivot table (below) does not include information on the category the athlete medaled in but does make it easier to see what metal medal each athlete received:
![image](https://github.com/user-attachments/assets/f36ffb80-a686-4091-a136-e89db9ada790)

## References 🧾:
Pandas Cheat Sheet: file:///C:/Users/amcco/Documents/Intro%20to%20DS/Pandas_Cheat_Sheet.pdf
Pie Chart Guide: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.pie.html
 - This helped me realize I needed numerical data for the pie chart
Tidy Data Frame: https://vita.had.co.nz/papers/tidy-data.pdf
Split method guide (never needed to use split): https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html

## Instructions 🏫:
Make sure to run from start to finish. I installed Matplotlib.pyplot, pandas, numpy, and seaborn but did not use the latter in my code.

## Features 🔥:
- You can see athletes sorted by the highest color medal they have won!
- The graphs I selected highlight the shortcomings of the dataset (only one medal and one category per person), which you can explore

Bonus pivot table: ![Pivoting Table](image.png)
