IMPORT Python3 AS Python;
IMPORT GNN.Tensor;
IMPORT Std.System.Log AS Syslog;
IMPORT Std;
IMPORT $.^.GAN_FL;
IMPORT $.^.Utils;
IMPORT GNN.GNNI;
IMPORT $.^.Types;
IMPORT GNN.Image;
IMPORT GNN.Types as GNNTypes;
IMPORT GNN.Utils as GNNUtils;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
FuncLayerDef := GNNTypes.FuncLayerDef;

#option('outputLimit',2000);
RAND_MAX := POWER(2,8) - 1;
RAND_MAX_2 := RAND_MAX / 2;

//Train data definitions
latentDim := 100;
batchSize := 100;
numEpochs := 1;
outputRows := 5;
outputCols := 5;
numRecords := 500;
numClasses := 10;

//Take MNIST dataset using GNN.Image module
mnist_train_images := Image.MNIST.Get_train_images('~mnist::images')[..numRecords];

//Take MNIST labels using GNN.Image module
mnist_train_labels  := Image.MNIST.Get_train_labels('mnist::labels')[..numRecords];

//Extract dimensions
imgDims := mnist_train_images[1].imgDims;
imgRows := imgDims[1];
imgCols := imgDims[2];
imgChannels := imgDims[3];

//Convert image to tensor with GNN.Image module
trainX_images := Image.ImgtoTens(mnist_train_images); 

//Convert labels to tensor
trainX_labels := NORMALIZE(mnist_train_labels, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id],
                            SELF.value := (REAL) (>UNSIGNED1<) LEFT.label));

//Call OneHotEncoder to get 10 output tensor
trainX_OH := GNNUtils.ToOneHot(trainX_labels, numClasses);
                                           
//Builds tensors for the neural network. Images will have wi = 1 and labels will have wi = 2. This should be in the order of inputs for the discriminator.
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, imgChannels], trainX_images) + Tensor.R4.MakeTensor([0, numClasses], trainX_OH, wi:=2); 

//Build a Y_train dataset
trainY := Tensor.R4.MakeTensor([0,1], DATASET(batchSize, TRANSFORM(TensData,
					SELF.indexes := [COUNTER, 1],
					SELF.value := 1)));

//GENERATOR
//Generator model definition information
ldef_generator :=  DATASET([{'input_g1','''layers.Input(shape=(100,))''',[]},
                            {'input_g2','''layers.Input(shape=(1,), dtype='int32')''',[]},
                            {'embed_g','''layers.Embedding(10,100)''',['input_g2']},
                            {'flatten_g','''layers.Flatten()''',['embed_g']},
                            {'multiply_g','''layers.Multiply()''',['input_g1','flatten_g']},
                            {'g1','''layers.Dense(256, input_dim=100)''', ['multiply_g']},
                            {'g2','''layers.LeakyReLU(alpha=0.2)''', ['g1']},
                            {'g3','''layers.BatchNormalization(momentum=0.8)''', ['g2']},
                            {'g4','''layers.Dense(512)''', ['g3']},
                            {'g5','''layers.LeakyReLU(alpha=0.2)''', ['g4']},
                            {'g6','''layers.BatchNormalization(momentum=0.8)''', ['g5']},
                            {'g7','''layers.Dense(1024)''', ['g6']},
                            {'g8','''layers.LeakyReLU(alpha=0.2)''', ['g7']},
                            {'g9','''layers.BatchNormalization(momentum=0.8)''', ['g8']},
                            {'g10','''layers.Dense(784, activation='tanh')''', ['g9']},
                            {'output_g','''layers.Reshape(28,28,1)''', ['g10']}],                
                        FuncLayerDef);

//DISCRIMINATOR
//Discriminator model definition information
ldef_discriminator := DATASET([{'input_d1','''layers.Input(shape=(28,28,1))''',[]},
                            {'input_d2','''layers.Input(shape=(1,), dtype='int32')''',[]},
                            {'embed_d','''layers.Embedding(10,784)''',['input_d2']},
                            {'flatten_embed','''layers.Flatten()''',['embed_d']},
                            {'flatten_image','''layers.Flatten()''',['input_d1']},
                            {'multiply_d','''layers.Multiply()''',['flatten_image','flatten_embed']},
                            {'d1','''layers.Dense(512, input_dim=784)''', ['multiply_d']},
                            {'d2','''layers.LeakyReLU(alpha=0.2)''', ['d1']},
                            {'d3','''layers.Dense(512)''', ['d2']},
                            {'d4','''layers.LeakyReLU(alpha=0.2)''', ['d3']},
                            {'d5','''layers.Dropout(0.4)''', ['d4']},
                            {'d6','''layers.Dense(512)''', ['d5']},
                            {'d7','''layers.LeakyReLU(alpha=0.2)''', ['d6']},
                            {'d8','''layers.Dropout(0.4)''', ['d7']},
                            {'output_d','''layers.Dense(1, activation='sigmoid')''', ['d8']}],                
                        FuncLayerDef);

//Compile string for both generator and discriminator
compiledef := '''compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';


//Get generator and discriminator models after training
myGAN := GAN_FL.Train(trainX, trainY(wi>=2), ldef_generator, ldef_discriminator, compiledef, batchSize, numEpochs, latentDim);

generator := myGAN.Generator;
discriminator := myGAN.Discriminator;

//Random set of normal data
random_data := DATASET(outputRows*OutputCols*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / RAND_MAX_2) - 1));

//Noise tensor
train_noise := Tensor.R4.MakeTensor([0,latentDim], random_data);

//Predict an image from noise
generated := GNNI.Predict(generator, train_noise);

//To get generator data as output
gen_data := Tensor.R4.GetData(generated);

//Outputs the tensor onto a file, so the graphs don't repeat
OUTPUT(gen_data, ,'~GAN::output_tensdata', OVERWRITE);

//Convert from tensor data to images by taking from file
outputImage := Image.TenstoImg(DATASET('~GAN::output_tensdata',TensData,FLAT ));

//Convert image data to jpg format to despray
mnistgrid := Image.OutputGrid(outputImage, outputRows, outputCols, 'Epoch_'+(String)numEpochs);

//Output the grid image to despray as a PNG using prefix filename,filesize
img_out := OUTPUT(mnistgrid, ,'~GAN::output_image', OVERWRITE);

//Despray variables
serv := 'server=http://192.168.56.101:8010 ';
over := 'overwrite=1 ';
action  := 'action=despray ';
dstip   := 'dstip=192.168.56.101 ';
dstfile := 'dstfile=/var/lib/HPCCSystems/mydropzone/*.png ';
srcname := 'srcname=~gan::output_image ';
splitprefix := 'splitprefix=filename,filesize ';
cmd := serv + over + action + dstip + dstfile + srcname + splitprefix;

//Despraying image onto landing zone
despray_image := STD.File.DfuPlusExec(cmd);
SEQUENTIAL(img_out, despray_image);

//Get the weights of the trained generator
weights := GNNI.GetWeights(generator);
/*
//Store the info of the model for predictions
modInfo := DATASET(1, TRANSFORM(Types.ModelInfo,
                            SELF.layerspec := Utils.makeLayerSpec(ldef_generator, compiledef),
                            SELF.modWeights := weights,
                            SELF.desprayCommand := cmd,
                            SELF.outputRows := outputRows,
                            SELF.outputCols := outputCols,
                            SELF.batchSize := batchSize,
                            SELF.latentDim := latentDim,
                            SELF.numEpochs := numEpochs
                            ));

OUTPUT(modInfo, ,'~GAN::GeneratorInfo', OVERWRITE);
*/