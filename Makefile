#======================================

# . . Define variables
dn		= /dev/null 


DAT=data
PAR=par
PKG=${DAT}/packages
PY=py
MK=mk

include ${MK}/build_packages.mk
include ${MK}/train_models.mk
