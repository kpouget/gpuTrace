#! /usr/bin/env python

#3997@compute_stacey_elastic_kernel<25,1,1><768,1,1>();
import fileinput #stdin or first arg
from collections import defaultdict

kernel_execs = defaultdict(int)

for line in fileinput.input():
    if not "@" in line: continue
    kernel = line.partition("@")[2].partition("<")[0]
    kernel_execs[kernel] += 1

print "{"
for kernel, count in kernel_execs.items():
    print '   {"%s", %d},' % (kernel, count)
print "   {NULL, NULL}"
print "}"

print "export LD_KERNEL_FILTER=" + ",".join(["{}:{}".format(kernel, count-1) for kernel, count in kernel_execs.items()])
