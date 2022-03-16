function [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ)
% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);
% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);
% Calculate the GAN loss
[lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated);
% For each network, calculate the gradients with respect to the loss.
gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);
end