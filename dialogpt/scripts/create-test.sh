#!/bin/bash

# Create input readable by run_gput2.py, and dup lines:

perl -ne '$line = $_; chomp $line; for (1..20) { print "$line\t__missing__\n"; }'
