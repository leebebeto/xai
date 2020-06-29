loss_type=$1
data_type=$2
image_type=$3
for (( c=0; c<=5; c++ ))
do
    echo $c
    bash train_script3.sh $c $loss_type $data_type $image_type
done

#bash train_script1.sh 0 both
#bash train_script1.sh 0 gan
#bash train_script1.sh 0 cycle
#bash train_script1.sh 1 both
#bash train_script1.sh 1 gan
#bash train_script1.sh 1 cycle
#bash train_script1.sh 2 both
#bash train_script1.sh 2 gan
#bash train_script1.sh 2 cycle
#bash train_script1.sh 3 both
#bash train_script1.sh 3 gan
#bash train_script1.sh 3 cycle
#bash train_script1.sh 4 both
#bash train_script1.sh 4 gan
#bash train_script1.sh 4 cycle
#bash train_script1.sh 5 both
#bash train_script1.sh 5 gan
#bash train_script1.sh 5 cycle

