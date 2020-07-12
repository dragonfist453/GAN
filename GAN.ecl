IMPORT Python3 AS Python;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Log AS Syslog;
IMPORT Std.System.Thorlib;
IMPORT Std.Str;
IMPORT $.Types;
IMPORT $.Utils;
nNodes := Thorlib.nodes();
nodeId := Thorlib.node();
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
LayerSpec := Types.LayerSpec;
/**
	* Generative Adversarial Networks
	* 
	* Generative Adversarial Networks are a pair of models which behave as adversaries to each other in order to compete
	* and learn from experience to generate artificial data. This data could be anything; audio, text, images, numerical data.
	* 
	* This module implements GANs using the Generalized Neural Network bundle. It can be used in various ways according to how 
	* the input is given and the output desired. Typically, the program flow goes as follows: -
	* 1) Read the dataset and convert into appropriate tensor using GNN.Tensor functions.
	* 2) Define the Generator and Discriminator model obeying the rules of GNN interface.
	* 3) Call GAN.train() with the parameters defined below to train the GANs for set number of Epochs and given batchSize
	* 4) Use returned generator to predict using GNN functions and returned discriminator to distinguish fake data from real data
	* 5) Output the predicted values as required for understanding
	*/
EXPORT GAN := MODULE																				
	/** The GAN module in the current bundle only has the train function which trains the GANs given the input, generator definition,
		* discriminator definition, batch size for each epoch and number of epochs. 
		* The training is carried by moving weights between the models after each Fit of a model so as to connect the models together.
		* This was very much required as the combined layer needs to be partially trained and contains a few non-trainable layers. In Python or JavaScript, 
		* the models are linked as objects, but here the linking was required to be done manually. 
		* For getting random sets of data from the input, a part of the tensor is extracted randomly as the dataset provided is assumed to be in no specific order.
		* It is advised to shuffle the dataset before giving to the train function.   
		* See Test/simpleGANtest.ecl to see the basic working of the GAN train. DCGAN is also implemented in Test/DCGANtest.ecl.
		* @param input The input tensor which contains the train dataset.
		* @param generator_ldef The generator layer definition as a set which may be passed for the making of generator model.
		* @param discriminator_ldef The discriminator layer definition as a set which may be passed for the making of discriminator model.
		* @param compiledef The compile string for both the generator and discriminator model so as to compile their losses, metrics and optimisers for training.
		* @param batchSize This is the batch size which is trained every epoch. batchSize number of records are used to train over.  
		* @param numEpochs This is the number of epochs for which the GAN model must train.
		* @param latentDim This is the latent dimension for noise, which is used by the generator. Default is set to 100.
		* @return A set containing the tokens of generator and discriminator model respectively.
		* Generator could be used to generate data. 
		* Discriminator could be used to distinguish fake data from real data due to the training received.
		*/
	EXPORT Train(DATASET(t_Tensor) input,
								SET OF STRING generator_ldef,
								SET OF STRING discriminator_ldef,
								STRING compiledef,
								UNSIGNED4 batchSize = 100,
								UNSIGNED4 numEpochs = 1,
								UNSIGNED4 latentDim = 100) := MODULE

		SET OF UNSIGNED4 doTrain := FUNCTION 

			//Limit for randomize function to be used when making noise
			RAND_MAX := POWER(2,8) - 1;
			RAND_MAX_2 := RAND_MAX / 2;

			//Gets the number of records in the tensor
			recordCount := TENSOR.R4.GetRecordCount(input);

			//Start session for GAN
			session := GNNI.GetSession();

			//Define generator network
			generator := GNNI.DefineModel(session, generator_ldef, compiledef); //Generator model definition

			//This is used to extract weights from combined and also merge weights back
			gen_wts_id := MAX(GNNI.GetWeights(generator), wi);

			//Define discriminator network
			discriminator := GNNI.DefineModel(session, discriminator_ldef, compiledef); //Discriminator model definition

			//Convert SET of strings to dataset to iterate over and make trainable false
			dis_temp := DATASET(discriminator_ldef[2..], {STRING layer});

			//Project this with a transform to remove suffix bracket and add non-trainable feature
			dis_notrain := PROJECT(dis_temp, TRANSFORM(RECORDOF(LEFT),
												SELF.layer := IF(Str.EndsWith(Str.RemoveSuffix(LEFT.layer, ')'), '('),  //If it ends with a ( after removing ), then the function has no parameter
																Str.RemoveSuffix(LEFT.layer, ')') + 'trainable=False)', //If no parameter, no comma
																Str.RemoveSuffix(LEFT.layer, ')') + ', trainable=False)') //If parameter exists, comma is required
											));

			//Get combined layers definition
			combined_ldef := generator_ldef + SET(dis_notrain, layer);

			//Define combined functional network
			combined := GNNI.DefineModel(session, combined_ldef, compiledef);

			//Dataset of 1s for classification
			//Repeated nNodes times for each node to process
			valid_data := DATASET(batchSize*nNodes, TRANSFORM(TensData,
					SELF.indexes := [COUNTER, 1],
					SELF.value := 1));
			valid := Tensor.R4.MakeTensor([0,1],valid_data);

			//Dataset of 0s for classification
			fake_data := DATASET(batchSize*nNodes, TRANSFORM(TensData,
					SELF.indexes := [COUNTER, 1],
					SELF.value := 0));
			fake := Tensor.R4.MakeTensor([0,1],fake_data);

			//Get only initial combined weights
			wts := GNNI.GetWeights(combined);

			//Fooling ECL to generate unique random datasets by passing unique integers which do nothing
			DATASET(TensData) makeRandom(UNSIGNED a) := FUNCTION
				reslt := DATASET(latentDim*batchSize*nNodes, TRANSFORM(TensData,
					SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
					SELF.value := ((RANDOM() % RAND_MAX) / RAND_MAX_2) - 1 * a / a));
				RETURN reslt;
			END; 

			//The loop that executes for training the GAN epochwise
			DATASET(t_Tensor) epochLoop(DATASET(t_Tensor) wts, UNSIGNED4 epochNum) := FUNCTION

				//Selecting random batch of images
				//Random position in Tensor which is (batchSize) less than COUNT(input)
				batchPos := RANDOM()%(recordCount/nNodes - batchSize);

				//Extract (batchSize) tensors starting from a random batchPos from the tensor input. Now we have a random input images of (batchSize) rows.
				X_dat := int.TensExtract(input, batchPos, batchSize);

				//Noise for generator to make fakes
				random_data1 := makeRandom(epochNum*2);
				train_noise1 := Tensor.R4.MakeTensor([0,latentDim], random_data1);

				//New model IDs
				loopDiscriminator := discriminator + 3*(epochNum - 1);
				loopCombined := combined + 2*(epochNum - 1);
				loopGenerator := generator + (epochNum - 1);

				//Split weights accordingly. Generator layer <= gen_wts_id. Discriminator layers > gen_wts_id. Discriminator must be subtracted by gen_wts_id to get its proper weights
				genWts := SORT(wts(wi <= (Tensor.t_WorkItem) gen_wts_id), wi, sliceid, LOCAL);
				splitdisWts := SORT(wts(wi > (Tensor.t_WorkItem) gen_wts_id), wi, sliceid, LOCAL);
				diswts := PROJECT(splitdisWts, TRANSFORM(t_Tensor,
					SELF.wi := LEFT.wi - gen_wts_id,
					SELF := LEFT
					));

				//Setting generator weights
				generator1 := GNNI.SetWeights(loopGenerator, genWts);

				//Predicting using Generator for fake images
				gen_X_dat := GNNI.Predict(generator1, train_noise1);

				//Setting discriminator weights
				discriminator1 := GNNI.SetWeights(loopDiscriminator, disWts); 

				//Fitting real data
				discriminator2 := GNNI.Fit(discriminator1, X_dat, valid, batchSize, 1);

				//Project generated data to get 0 in first shape component
				generated_dat := PROJECT(gen_X_dat, TRANSFORM(t_Tensor,
					SELF.shape := [0] + LEFT.shape[2..],
					SELF := LEFT
					));

				//Fitting generated data
				discriminator3 := GNNI.Fit(discriminator2, generated_dat, fake, batchSize, 1);

				//Noise to train combined model
				random_data2 := makeRandom(epochNum*2 + 1);
				train_noise2 := Tensor.R4.MakeTensor([0,latentDim], random_data2);

				//Get discriminator weights, add 20 to it, change discriminator weights of combined model, set combined weights
				updateddisWts := GNNI.GetWeights(discriminator3);
				newdisWts := PROJECT(updateddisWts, TRANSFORM(t_Tensor,
									SELF.wi := LEFT.wi + gen_wts_id,
									SELF := LEFT
									));
				comWts := SORT(wts(wi <= (Tensor.t_WorkItem) gen_wts_id) + newdisWts(wi > (Tensor.t_WorkItem) gen_wts_id), wi, sliceid, LOCAL);
				combined1 := GNNI.SetWeights(loopCombined, comWts);

				//Fit combined model
				combined2 := GNNI.Fit(combined1, train_noise2, valid, batchSize, 1);

				//Get combined weights to return
				newWts := GNNI.GetWeights(combined2);
				
				//Logging progress when done
				logProgress := Syslog.addWorkunitInformation('GAN training - Epoch : '+epochNum);

				//Log progress if the newWts are produced
				RETURN WHEN(newWts, logProgress);
			END;        

			//Call loop to train numEpochs times
			finalWts := LOOP(wts, ROUNDUP(numEpochs), epochLoop(ROWS(LEFT),COUNTER));

			//Final model IDs
			finalGenerator := generator + numEpochs + 1;
			finalDiscriminator := discriminator + numEpochs + 1;

			//Setting new weights of generator
			genWts := SORT(finalWts(wi <= gen_wts_id), wi, sliceid, LOCAL);
			generator_trained := GNNI.SetWeights(finalGenerator, genWts);

			//Setting new weights of discriminator
			splitdisWts := SORT(wts(wi > gen_wts_id), wi, sliceid, LOCAL);
			diswts := PROJECT(splitdisWts, TRANSFORM(t_Tensor,
				SELF.wi := LEFT.wi - gen_wts_id,
				SELF := LEFT
				));
			discriminator_trained := GNNI.SetWeights(finalDiscriminator, disWts);

			//Return the generator id to use generator to predict
			RETURN [generator_trained, discriminator_trained];
		END;

		SHARED modelIds := doTrain;
		EXPORT Generator := modelIds[1];
		EXPORT Discriminator := modelIds[2];
	END; 
END;	