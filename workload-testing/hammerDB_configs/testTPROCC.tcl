dbset db db2

# Configure TPC-C workload parameters
diset tpcc db2_count_ware 100          ;# Set number of warehouses
diset tpcc db2_num_vu 100              ;# Set number of virtual users
diset tpcc db2_rampup 1                ;# Ramp-up time in minutes
diset tpcc db2_duration 60             ;# Duration in minutes
diset tpcc db2_total_iterations 1000000 ; # Total iterations
diset tpcc db2_async_scale true        ;# Enable asynchronous scaling
diset tpcc db2_async_client 10         ;# Number of asynchronous clients
diset tpcc db2_async_delay 0           ;# Delay between transactions (0 = no delay)

# Remove invalid key (runtofinish)
# RunToFinish is not a valid key in Db2, so it is removed

# Debugging (optional)
puts "Starting TPC-C workload for Db2..."

# Create virtual users and start the workload
vucreate
vurun