
ConvNet2D_A_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_A.json
	$< \
	--json_params ${PAR}/ConvNet2D_A.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_A

ConvNet2D_B_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_B.json
	$< \
	--json_params ${PAR}/ConvNet2D_B.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_B

ConvNet2D_B_20240319_225105_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_B.json
	$< \
	--json_params ${PAR}/ConvNet2D_B.json \
	--train_data ${PKG}/20240319_225105_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_225105_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_225105_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_225105_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_B
