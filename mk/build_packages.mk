build_data_packages: ${PY}/build_data_packages.py ${PAR}/data_defaults.json
	$< -json_params ${PAR}/data_defaults.json
