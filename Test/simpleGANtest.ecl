IMPORT Python3 AS Python;
IMPORT GNN.Tensor;
IMPORT Std.System.Thorlib;
IMPORT Std.System.Log AS Syslog;
IMPORT Std;
IMPORT IMG.IMG;
IMPORT GAN.GAN;
IMPORT GAN.Utils;
IMPORT GNN.GNNI;
IMPORT GAN.Types;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

#option('outputLimit',2000);
RAND_MAX := POWER(2,8) - 1;
RAND_MAX_2 := RAND_MAX / 2;

//Train data definitions
imgRows := 28;
imgCols := 28;
imgChannels := 1;
imgSize := imgRows * imgCols;
latentDim := 100;
batchSize := 100;
numEpochs := 2000;
outputRows := 5;
outputCols := 5;
numRecords := 60000;

//Take MNIST dataset using IMG module
mnist_train_images := IMG.MNIST_train_image()[..numRecords];

//Tensor dataset having image data normalised to range of -1 to 1
trainX0 := NORMALIZE(mnist_train_images, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28+1, (COUNTER-1)%28+1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 )); 
                     
//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, 1], trainX0); 

//GENERATOR
//Generator model definition information
ldef_generator := ['''layers.Input(shape=(100,))''',
                    '''layers.Dense(256, input_dim=100)''',
                    '''layers.LeakyReLU(alpha=0.2)''',    
                    '''layers.BatchNormalization(momentum=0.8)''',
                    '''layers.Dense(512)''',
                    '''layers.LeakyReLU(alpha=0.2)''',
                    '''layers.BatchNormalization(momentum=0.8)''',
                    '''layers.Dense(1024)''',
                    '''layers.LeakyReLU(alpha=0.2)''',
                    '''layers.BatchNormalization(momentum=0.8)''',
                    '''layers.Dense(784,activation="tanh")''',
                    '''layers.Reshape((28,28,1))'''];

//DISCRIMINATOR
//Discriminator model definition information
ldef_discriminator := ['''layers.Input(shape=(28,28,1))''',
                        '''layers.Flatten(input_shape=(28,28,1))''',
                        '''layers.Dense(512)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.Dense(256)''',
                        '''layers.LeakyReLU(alpha=0.2)''',
                        '''layers.Dense(1,activation="sigmoid")'''];

//Compile string for both generator and discriminator
compiledef := '''compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';


//Get generator and discriminator models after training
myGAN := GAN.Train(trainX, ldef_generator, ldef_discriminator, compiledef, batchSize, numEpochs, latentDim);

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

//To make up for multiple images output
gen_data := IMG.GenCorrect(generated);

//Outputs the tensor onto a file, so the graphs don't repeat
OUTPUT(gen_data, ,'~GAN::output_tensdata', OVERWRITE);

//Convert from tensor data to images by taking from file
outputImage := IMG.TenstoImg(DATASET('~GAN::output_tensdata',TensData,FLAT ));

//Convert image data to jpg format to despray
mnistgrid := IMG.OutputGrid(outputImage, outputRows, outputCols, numEpochs);

//Output the grid image to despray as a PNG using prefix filename,filesize
img_out := OUTPUT(mnistgrid, ,'~GAN::output_image', OVERWRITE);

//Despray variables
serv := 'server=localhost:8010 ';
over := 'overwrite=1 ';
action  := 'action=despray ';
dstip   := 'dstip=192.168.86.149 ';
dstfile := 'dstfile=/var/lib/HPCCSystems/mydropzone/*.png ';
srcname := 'srcname=~gan::output_image ';
splitprefix := 'splitprefix=filename,filesize ';
cmd := serv + over + action + dstip + dstfile + srcname + splitprefix;

//Despraying image onto landing zone
despray_image := STD.File.DfuPlusExec(cmd);
SEQUENTIAL(img_out, despray_image);

//Get the weights of the trained generator
weights := GNNI.GetWeights(generator);

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