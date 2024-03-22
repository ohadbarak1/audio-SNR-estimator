#---------------------------------------------
# run workflows with Mean Absolute Error loss
#---------------------------------------------
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
	${PKG}/20240319_214050_10000_files/ConvNet2D_A_bb47014c0210bbb2/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_A.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_A_bb47014c0210bbb2/model.h5

ConvNet2D_A_nonorm_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_A_nonorm.json
	$< \
	--json_params ${PAR}/ConvNet2D_A_nonorm.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_A_nonorm

Infer_test_ConvNet2D_A_nonorm_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_A.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_A_nonorm_/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_A.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_A_nonorm_/model.h5

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
	${PKG}/20240319_214050_10000_files/ConvNet2D_B_43453041f8d4f053/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_B.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_B_43453041f8d4f053/model.h5

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


#---------------------------------------------
# run workflows with Mean Squared Error loss
#---------------------------------------------
ConvNet2D_Amse_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_Amse.json
	$< \
	--json_params ${PAR}/ConvNet2D_Amse.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_Amse

Infer_test_ConvNet2D_Amse_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_Amse.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_Amse_ae0e6b36e47eb141/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_Amse.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_Amse_ae0e6b36e47eb141/model.h5

ConvNet2D_Bmse_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_Bmse.json
	$< \
	--json_params ${PAR}/ConvNet2D_Bmse.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_Bmse

Infer_test_ConvNet2D_Bmse_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_Bmse.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_Bmse_3e869235c76eb293/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_Bmse.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_Bmse_3e869235c76eb293/model.h5

ConvNet2D_Cmse_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_Cmse.json
	$< \
	--json_params ${PAR}/ConvNet2D_Cmse.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_Cmse

Infer_test_ConvNet2D_Cmse_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_Cmse.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_Cmse_3a0bdf641aac10a8/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_Cmse.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_Cmse_3a0bdf641aac10a8/model.h5

#---------------------------------------------
# run workflows with Mean Squared Error loss and no normalization
#---------------------------------------------
ConvNet2D_Cmse_nonorm_20240319_214050_10000_files:  ${PY}/train_model.py ${PAR}/ConvNet2D_Cmse.json
	$< \
	--json_params ${PAR}/ConvNet2D_Cmse_nonorm.json \
	--train_data ${PKG}/20240319_214050_10000_files/train_data.npy \
	--train_labels ${PKG}/20240319_214050_10000_files/train_labels.npy \
	--valid_data ${PKG}/20240319_214050_10000_files/valid_data.npy \
	--valid_labels ${PKG}/20240319_214050_10000_files/valid_labels.npy \
	--model_pre ConvNet2D_Cmse_nonorm

Infer_test_ConvNet2D_Cmse_nonorm_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_Cmse.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_Cmse_nonorm_1346386b034b5652/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_Cmse_nonorm.json \
	--test_data ${PKG}/20240319_214050_10000_files/test_data.npy \
	--test_labels ${PKG}/20240319_214050_10000_files/test_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_Cmse_nonorm_1346386b034b5652/model.h5



#------------------------------------------
# run inference on evaluation data
#------------------------------------------

Infer_eval_ConvNet2D_C_20240319_214050_10000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_C.json \
	${PKG}/20240319_214050_10000_files/ConvNet2D_C_89fea9e5b333a930/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_C.json \
	--test_data ${PKG}/20240321_105750_20000_files/eval_data.npy \
	--test_labels ${PKG}/20240321_105750_20000_files/eval_labels.npy \
	--model_path ${PKG}/20240319_214050_10000_files/ConvNet2D_C_89fea9e5b333a930/model.h5


Infer_eval_ConvNet2D_C_20240320_163218_100000_files: \
	${PY}/infer.py \
	${PAR}/ConvNet2D_C.json \
	${PKG}/20240320_163218_100000_files/ConvNet2D_C_cedc43e341bfa311/model.h5

	$< \
	--json_params ${PAR}/ConvNet2D_C.json \
	--test_data ${PKG}/20240321_105750_20000_files/eval_data.npy \
	--test_labels ${PKG}/20240321_105750_20000_files/eval_labels.npy \
	--model_path ${PKG}/20240320_163218_100000_files/ConvNet2D_C_cedc43e341bfa311/model.h5

