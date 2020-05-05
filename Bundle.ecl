IMPORT Std;
EXPORT Bundle := MODULE(Std.BundleBase)
  EXPORT Name := 'GAN';
  EXPORT Description := 'Generative Adversarial Neural Networks Bundle';
  EXPORT Authors := ['HPCCSystems'];
  EXPORT License := 'See LICENSE.TXT';
  EXPORT Copyright := 'Copyright (C) 2020 HPCC Systems®';
  EXPORT DependsOn := ['GNN 1.1', 'ML_Core'];
  EXPORT Version := '1.0';
  EXPORT PlatformVersion := '7.4.0';
END;