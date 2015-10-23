#!/bin/sh
gcc -shared -Wl,-install_name,rbmball.so -o rbmball.so -fPIC rbmball.c normal.c