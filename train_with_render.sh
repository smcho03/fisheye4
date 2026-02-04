
#!/bin/bash



MODEL_PATH="./output/sport_150"

DATA_PATH="~/UROP/UROP_csm/fisheye/data/sportunifront"



echo "🚀 Starting training..."



# 백그라운드로 학습 시작

python train.py \

  -s $DATA_PATH \

  -m $MODEL_PATH \

  --eval \

  -r 1 \

  --save_iterations 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000 30000 \

  --test_iterations 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000 30000 &



TRAIN_PID=$!

echo "Training PID: $TRAIN_PID"



# 1500마다 렌더링

for iter in 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000; do

    echo "⏳ Waiting for iteration $iter checkpoint..."

    

    # Checkpoint 파일 생성될 때까지 대기

    while [ ! -f "$MODEL_PATH/point_cloud/iteration_$iter/point_cloud.ply" ]; do

        if ! ps -p $TRAIN_PID > /dev/null; then

            echo "❌ Training process stopped!"

            exit 1

        fi

        sleep 30

    done

    

    echo "🎨 Rendering iteration $iter..."

    python render.py -m $MODEL_PATH --iteration $iter --skip_train

    

    echo "✅ Iteration $iter rendered!"

done



# 최종 30000 렌더링은 학습 완료 후

wait $TRAIN_PID

echo "🎨 Rendering final iteration 30000..."

python render.py -m $MODEL_PATH --iteration 30000 --skip_train



echo "🎉 All done!"

