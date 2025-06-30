dbset db db2
print dict
diset connection db2_def_user db2admin
diset connection db2_def_pass root
diset connection db2_def_dbase testdb3
diset tpcc db2_dbase testdb3
diset tpcc db2_user db2admin
diset tpcc db2_pass root


diset tpcc db2_count_ware 10
diset tpcc db2_num_vu 10
diset tpcc db2_rampup 2
diset tpcc db2_duration 10
 
buildschema
 