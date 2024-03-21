build_data_packages: ${PY}/build_data_packages.py ${PAR}/data_defaults.json
	$< -json_params ${PAR}/data_defaults.json

build_data_packages_100K: ${PY}/build_data_packages.py ${PAR}/data_defaults_100K.json
	$< -json_params ${PAR}/data_defaults_100K.json

build_test_data_packages_20K: ${PY}/build_data_packages.py ${PAR}/test_data_20K.json
	$< -json_params ${PAR}/test_data_20K.json
