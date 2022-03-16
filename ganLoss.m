function [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated)
% Calculate losses for the discriminator network.
lossGenerated = -mean(log(1-sigmoid(dlYPredGenerated)));
lossReal = -mean(log(sigmoid(dlYPred)));
% Combine the losses for the discriminator network.
lossDiscriminator = lossReal + lossGenerated;
% Calculate the loss for the generator network.
lossGenerator = -mean(log(sigmoid(dlYPredGenerated)));
end