# RARE-X: Sample Python Model

## Overview
This repository contains a sample model for Task 2 of the [RARE-X: A Rare Disease Open Science Data Challenge](https://www.synapse.org/rarex).

## Model Description
This model is the containerized version of the [provided notebook](https://www.synapse.org/#!Synapse:syn51942435). The model follows the TPOT pipeline:

1. add/remove features
2. impute missing values
3. apply other transforms
4. perform Random Forest Classifier

| ![TPOT pipeline](https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-pipeline-example.png) |
|:--:|
| _Source: http://epistasislab.github.io/tpot/_|

## Build the Model
1. Replace the TPOT pipeline with your own model!

2. Update `requirements.txt` as needed.

3. Dockerize the model:

   ```
   docker build -t docker.synapse.org/<project id>/my-model:v1 .
   ```

   where:
   * `<project id>`: Synapse ID of your project
   * `my-model`: name of your model
   * `v1`: version of your model
   * `.`: filepath to the Dockerfile

4. (optional but recommended) Locally run the model to ensure it can run successfully. For this, you may use [dummy_task2](https://www.synapse.org/#!Synapse:syn51614785) and [dummy_task2_test](https://www.synapse.org/#!Synapse:syn51974898) as the mounts for `/input` and `/test`, respectively. E.g.

   ```
   docker run --rm \
     --network none \
     --volume /path/to/dummy_task2:/input:ro \
     --volume /path/to/dummy_task2_test:/test:ro \
     --volume /path/to/output:/output:rw \
     docker.synapse.org/<project id>/my-model:v1
   ```

5. Use `docker push` to push the model up to your project on Synapse, then submit it to the challenge.

For more information on how to submit, refer to the [Submission Tutorial](https://www.synapse.org/#!Synapse:syn51198355/wiki/622697) on the challenge site.

## Credit
**Author**:

Jake Albrecht (@chepyle)

**Contributors**:
* Maria Diaz (@mdsage1)
* Verena Chung (@vpchung)
