echo "Hello World !\n"
echo "please choose which model to use\n"
echo "basic CNN :Net1(0),Net2(1),Net3(2)\n"
echo "ResNet: ResNet18(3) ResNet34(4) ResNet50(5) ResNet101(6) ResNet152(7)\n"
echo "AlexNet: AlexNet(8)\n"



echo "The default model is Net1\n"

echo "batch_size:\n  The default batch_size is 32\n"

ls
python train.py $1 $2
echo "The train has been done!\n"
python evaluate.py $1 $2
echo "Start the test!\n"
python test.py
echo "Start to draw!\n"
python draw.py $1 $2


