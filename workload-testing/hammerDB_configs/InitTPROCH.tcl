dbset db db2
print dict
diset connection db2_def_user db2admin
diset connection db2_def_pass root
diset connection db2_def_dbase testDB5
diset tpch db2_dbase testDB5
diset tpch db2_user db2admin
diset tpch db2_pass root

diset tpch db2_scale_factor 5   # Set the scale factor (1 = 1 GB)
diset tpch db2_num_vu 1         # Number of virtual users
diset tpch db2_rampup 1         # Ramp-up time in minutes
diset tpch db2_duration 1       # Test duration in minutes

buildschema     