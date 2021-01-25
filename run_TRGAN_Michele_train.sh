# Setup the environment
source /home/node2/anaconda3/etc/profile.d/conda.sh
conda activate tf2.3

TASK="TRGAN"
MODE="train"
BATCH_SIZE=128
NEI=64
EKW=1
ESW=1
DKW=17
DSW=1
NGO=48
NDI=32
LEARNING_RATE=0.00002
MAX_EPOCH=500
ADVERSARIAL_RATIO=1.e-1
LAMBDA_VEL_GRAD=0
OVERLAPPRED=0
SUFFIX="_Michele_NEI${NEI}_EKW${EKW}_ESW${ESW}_DKW${DKW}_DSW${DSW}_NGO${NGO}_NDI${NDI}_LR${LEARNING_RATE}_AR${ADVERSARIAL_RATIO}_VG${LAMBDA_VEL_GRAD}_OL${OVERLAPPRED}"

CMD="python main.py \
--task ${TASK} \
--batch_size ${BATCH_SIZE} \
--nei ${NEI} \
--ekW ${EKW} \
--esW ${ESW} \
--dkW ${DKW} \
--dsW ${DSW} \
--ngo ${NGO} \
--ndi ${NDI} \
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
