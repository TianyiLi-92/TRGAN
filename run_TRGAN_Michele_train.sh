# Setup the environment
source /home/node2/anaconda3/etc/profile.d/conda.sh
conda activate tf2.3

TASK="TRGAN"
MODE="train"
BATCH_SIZE=128
NEI=64
KW=1
SW=1
NGO=32
conditionAdv=True
LEARNING_RATE=0.00002
MAX_EPOCH=500
ADVERSARIAL_RATIO=1.e-1
LAMBDA_VEL_GRAD=0
OVERLAPPRED=0
SUFFIX="_Michele_NEI${NEI}_KW${KW}_SW${SW}_NGO${NGO}_CA_LR${LEARNING_RATE}_AR${ADVERSARIAL_RATIO}_VG${LAMBDA_VEL_GRAD}_OL${OVERLAPPRED}"

CMD="python main.py \
--task ${TASK} \
--batch_size ${BATCH_SIZE} \
--nei ${NEI} \
--kW ${KW} \
--sW ${SW} \
--ngo ${NGO} \
--conditionAdv ${conditionAdv} \
--learning_rate ${LEARNING_RATE} \
--max_epoch ${MAX_EPOCH} \
--adversarial_ratio ${ADVERSARIAL_RATIO} \
--lambda_vel_grad ${LAMBDA_VEL_GRAD} \
--overlapPred ${OVERLAPPRED} \
--output_dir ./output/${TASK}${SUFFIX} \
--summary_dir ./summary/${TASK}${SUFFIX} \
--mode ${MODE} "


echo ${CMD} > ./logs/${TASK}_${MODE}${SUFFIX}_log.out

${CMD} >> ./logs/${TASK}_${MODE}${SUFFIX}_log.out
