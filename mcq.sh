#!/bin/sh
gcc -shared -Wl,-install_name,mcq.so -o mcq.so -fPIC mcq.c normal.c