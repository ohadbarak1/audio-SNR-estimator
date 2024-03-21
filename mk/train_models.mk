
ConvNet2D_A_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_A.json
	$< \
	--json_params ${PAR}/ConvNet2D_A.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_A

Infer_test_ConvNet2D_A_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_A.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_A_a4aa203ebdb30f26/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_A.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_A_a4aa203ebdb30f26/model.h5

ConvNet2D_B_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_B.json
	$< \
	--json_params ${PAR}/ConvNet2D_B.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_B

Infer_test_ConvNet2D_B_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_B.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_B_0888e88b360f7eb0/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_B.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_B_0888e88b360f7eb0/model.h5

ConvNet2D_C_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_C.json
	$< \
	--json_params ${PAR}/ConvNet2D_C.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_C

Infer_test_ConvNet2D_C_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_C.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_C_89fea9e5b333a930/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_C.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_C_89fea9e5b333a930/model.h5

ConvNet2D_C_20240320_163218_100000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_C.json
	$< \
	--json_params ${PAR}/ConvNet2D_C.json \
	--train_data ${PKG}/20240320_163218_100000_files/train_data.npy \
	--train_labels ${PKG}/20240320_163218_100000_files/train_labels.npy \
	--valid_data ${PKG}/20240320_163218_100000_files/valid_data.npy \
	--valid_labels ${PKG}/20240320_163218_100000_files/valid_labels.npy \
	--model_pre ConvNet2D_C

Infer_test_ConvNet2D_C_20240320_163218_100000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_C.json \
	${PKG}/20240320_163218_100000_files/ConvNet2D_C_cedc43e341bfa311/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_C.json \
	--test_data ${PKG}/20240320_163218_100000_files/test_data.npy \
	--test_labels ${PKG}/20240320_163218_100000_files/test_labels.npy \
	--model_path ${PKG}/20240320_163218_100000_files/ConvNet2D_C_cedc43e341bfa311/model.h5


