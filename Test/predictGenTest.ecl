IMPORT GNN.GNNI;
IMPORT GNN.Image;
IMPORT GNN.Tensor;
IMPORT GAN.Types;
IMPORT STD;
TensData := Tensor.R4.TensData;
t_Tensor := Tensor.R4.t_Tensor;
RAND_MAX := POWER(2,8) - 1;
RAND_MAX_2 := RAND_MAX / 2;


savedModel := DATASET('~DCGAN::GeneratorInfo',Types.ModelInfo,FLAT );
savedlayerDef := savedModel[1].layerspec[1].layerDef;
savedCompilestr := savedModel[1].layerspec[1].compileDef;
savedWeights := savedModel[1].modWeights;
outputRows := savedModel[1].outputRows;
outputCols := savedModel[1].outputCols;
batchSize := savedModel[1].batchSize;
latentDim := savedModel[1].latentDim;
numEpochs := savedModel[1].numEpochs;
cmd := savedModel[1].desprayCommand;

session := GNNI.GetSession();

savedGenerator := GNNI.DefineModel(session, savedlayerDef, savedCompilestr);
generator := GNNI.SetWeights(savedGenerator, savedWeights);

//Random set of normal data
random_data := DATASET(outputRows*OutputCols*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / RAND_MAX_2) - 1));

//Noise tensor
train_noise := Tensor.R4.MakeTensor([0,latentDim], random_data);

//Predict an image from noise
generated := GNNI.Predict(generator, train_noise);

//To make up for multiple images output
gen_data := Tensor.R4.GetData(generated);

//Convert from tensor data to images by taking from file
outputImage := Image.TenstoImg(gen_data);

//Convert image data to jpg format to despray
mnistgrid := Image.OutputGrid(outputImage, outputRows, outputCols, 'Epoch_'+(STRING)numEpochs);

//Output the grid image to despray as a PNG using prefix filename,filesize
img_out := OUTPUT(mnistgrid, ,'~GAN::output_image', OVERWRITE);

//Despraying image onto landing zone
despray_image := STD.File.DfuPlusExec(cmd);
SEQUENTIAL(img_out, despray_image);