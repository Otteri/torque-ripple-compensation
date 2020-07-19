#ifndef QLEARNING_H
#define QLEARNING_H

struct Maximum {
    uint16_t idx;
    float value;
};

class Qtable {
#define ANGLE_NUM 100 // These must match with the qtable!
#define ACTION_NUM 7  // The table size needs to be known beforehandedly
#define T_MAX 0.12    // for memory allocation, hence #defines.
typedef float qmatrix[ANGLE_NUM][ACTION_NUM];

public:
    Qtable(float alpha, float gamma, float e);
    bool loadTable();
    void clearTable(); // zeroes weights
    float getBestAction(float current_angle); // Returns the best known action
    float train(float angle, float actual, float reference);

    uint32_t train_iterations; // how long should train?
    bool is_learning;
    float reward; // for monitoring
    float action;

    // Hyperparameters:
    float epsilon;
    float alpha;
    float gamma;
    float ek;
    float lambda;

private:
    float getReward(float actual, float reference);
    uint16_t findClosestIdx(float arr[], uint16_t n, float target);
    uint16_t getCloserIdx(float arr[], uint16_t idx1, uint16_t idx2, float target);
    struct Maximum findMax(float* p_weights);
    void linspaceAngles(float result[], float min, float max);
    void linspaceActions(float result[], float min, float max);
    float getRandom();
    uint16_t getRandomInteger(uint16_t min, uint16_t max);
    void resetState();
    void copyWeights(float* table1, float* table2);
    void hasFinishedTraining();
    void update(float actual, uint16_t angle_idx);
    void updateTargetTable(bool has_improved);
    bool updateRewardAverage(float reward);
    void dumpTable();
    bool hasImproved(float actual, bool reset);

    float rotation_min;
    float rotation_max;
    float ripple_min;

    // Table itself
    float* qtable_ptr; // points to the first item
    qmatrix* p_qtable;
    float* qtable_target_ptr;
    float qtable_target_weights[ANGLE_NUM][ACTION_NUM];

    // Arrays used for converting weights to something sensible
    float angles[ANGLE_NUM];
    float actions[ACTION_NUM];

    // The previous values must be kept in memory
    // For the table update.
    uint16_t last_angle_idx;
    uint16_t last_action_idx;
    uint32_t iteration_number;

    // For finding the best weights
    float cumulative_reward;
    float average_reward;
    float max_average_reward;
    bool auto_zeta_search;

    uint16_t N; // average over N-numbers and increment zeta every-N
    uint8_t electrical_period_count;
    bool is_full_rotation;
    bool save;
};
#endif
