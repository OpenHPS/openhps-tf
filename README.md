<h1 align="center">
  <img alt="OpenHPS" src="https://openhps.org/images/logo_text-512.png" width="40%" /><br />
  @openhps/tf
</h1>
<p align="center">
    <a href="https://github.com/OpenHPS/openhps-tf/actions/workflows/main.yml" target="_blank">
        <img alt="Build Status" src="https://github.com/OpenHPS/openhps-tf/actions/workflows/main.yml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/OpenHPS/openhps-tf">
        <img src="https://codecov.io/gh/OpenHPS/openhps-rf/branch/master/graph/badge.svg"/>
    </a>
    <a href="https://codeclimate.com/github/OpenHPS/openhps-tf/" target="_blank">
        <img alt="Maintainability" src="https://img.shields.io/codeclimate/maintainability/OpenHPS/openhps-tf">
    </a>
    <a href="https://badge.fury.io/js/@openhps%2Ftf">
        <img src="https://badge.fury.io/js/@openhps%2Ftf.svg" alt="npm version" height="18">
    </a>
</p>

<h3 align="center">
    <a href="https://github.com/OpenHPS/openhps-core">@openhps/core</a> &mdash; <a href="https://openhps.org/docs/tf">API</a>
</h3>

<br />

## Getting Started
If you have [npm installed](https://www.npmjs.com/get-npm), start using @openhps/tf with the following command.
```bash
npm install @openhps/tf --save
```

## Usage

The repo contains the OpenHPS TensorFlow integration (services/nodes) and the
**accuracy-model experiment** — an AI model that predicts kNN fingerprinting
error in metres. Full architecture, the changes made, and measured results are
in **[docs/MODEL.md](docs/MODEL.md)**.

### The experiment

```bash
npm install
# Train + evaluate the accuracy model on the bundled OpenHPS-2021-05 data:
EPOCHS=250 PATIENCE=20 BATCH_SIZE=256 npm run test -- --grep "data.openhps.accuracy"
```

The suite (a) trains an autoencoder + accuracy head + radio-map model,
(b) evaluates how close the predicted error is to the true kNN error per `k`,
and (c) writes `accuracy_results.json` (heatmap source). Configuration via env:
`ENSEMBLE` (train N models and average predictions — the one measured
improvement, ~6–8 % lower error), `MODEL_DIR`.

### Remote GPU training (srv3)

Training runs on the remote box (`192.168.1.5`) in Docker — no system CUDA or
Node needed on the host. Copy `.env.example` to `.env` and adjust, then:

```bash
./scripts/sync.sh            # push source + data (node_modules stays remote)
./scripts/remote_setup.sh    # one-time: verify GPU + build the training image
./scripts/train_remote.sh    # detached run, tails logs, pulls model back
```

The training container uses `tfjs-node-gpu`, which requires an Ampere (sm_86+)
GPU.

## Contributors
The framework is open source and is mainly developed by PhD Student Maxim Van de Wynckel as part of his research towards *Hybrid Positioning and Implicit Human-Computer Interaction* under the supervision of Prof. Dr. Beat Signer.

Maintained by [MVdW-Software](https://mvdw-software.com).

## Contributing
Use of OpenHPS, contributions and feedback is highly appreciated. Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

## License
Copyright (C) 2019-2024 Maxim Van de Wynckel & Vrije Universiteit Brussel

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
