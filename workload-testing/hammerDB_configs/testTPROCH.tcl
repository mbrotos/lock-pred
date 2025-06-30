dbset db db2
diset tpch db2_num_vu 5           ;# Increase the number of virtual users to 100
diset tpch db2_scale_factor 5      ;# Increase the scale factor to 50 (equivalent to warehouses in TPC-C)
diset tpch db2_rampup 5            ;# Increase ramp-up time to 5 minutes
diset tpch db2_duration 60         ;# Increase duration to 60 minutes
diset tpch db2_async_scale true
diset tpch db2_async_client 5      ;# Increase the number of async clients to 1000
diset tpch db2_async_delay 10      ;# Decrease the delay to 10 ms to create more contention
diset tpch runtofinish true        ;# Ensure the test runs to completion

vucreate
vurun
