insert overwrite directory 'Che4ako_hiveout'
row format delimited fields terminated by '\t' lines terminated by '\n'
stored as textfile
select * from hw2_pred;
