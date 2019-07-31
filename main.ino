#include <ArduinoJson.h>
#include <stdio.h>

#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

int freeMemory() {
  char top;
#ifdef __arm__
  return &top - reinterpret_cast<char*>(sbrk(0));
#elif defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
  return &top - __brkval;
#else  // __arm__
  return __brkval ? &top - __brkval : &top - __malloc_heap_start;
#endif  // __arm__
}

bool debug = true;

void info(String a){
  if(debug){
    Serial.println(a);
  }
}

const long BAUD = 115200;

class SerialReader {
    char *data = NULL;
    String str_data = "";
    int state = 0;
    int size_data = -1;
    int digits_number = -1;
    char *message_size = NULL;
    long idx_digits = 0;
    long idx_data = 0;

public:

    SerialReader(){

    }

    void reset() {
        //Serial.println("Reseting SerialReader");
        free(data);
        data=NULL;
        size_data = -1;
        digits_number = -1;
        free(message_size);
        message_size=NULL;
        idx_digits = 0;
        idx_data = 0;
        state = 0;
    }

    String loop() {
        String str_data="";
        int error = read_serial();
        switch (error){
          case 0:
            info("Success !!");
            str_data = String(data);
            reset();
            //info("data=|"+str_data+"|");
            break;
          case 1:
            str_data = "";
            break;
          case 2:
            str_data = "";
            Serial.println("Malformed message, missing >");
            reset();
            break;
          default:
            str_data = "";
            reset();
            break;
        }


        return str_data;
    }

private:
    int read_serial() {
        char startMarker = '<';
        char endMarker = '>';
        char rc;
        while (Serial.available() > 0) {
            rc = Serial.read();
            //Serial.println("rc="+String(rc));
            if (rc == startMarker) {
               info("Marker '<' spotted, starting reading");
               reset();
               if (state!=0){
                  info("[WARN] starting a new reading, previous reading unfinished");
               }
               state = 1;
               info("Switching to state 1");
            } else if (state == 1) {
                digits_number = rc - '0';
                message_size = (char *) malloc(sizeof(char)*(digits_number + 1));
                state = 2;
                info("Switching to state 2");
            } else if (state == 2) {
                message_size[idx_digits] = rc;
                idx_digits++;
                if (idx_digits == digits_number) {
                    message_size[digits_number] = '\0';
                    size_data = atoi(message_size);
                    data = (char *) malloc(sizeof(char)*(size_data + 1));
                    state = 3;
                    info("Switching to state 3");
                }
            } else if (state == 3) {
                data[idx_data] = rc;
                idx_data++;
                if (idx_data == size_data) {
                    data[size_data] = '\0';
                    state = 4;
                    info("Switching to state 4");
                }
            }else if(state==4){
              if (rc == endMarker){
                info("endMarker spotted");
                state=0;
                return 0;
              }else{
                info("No endMarker while state 4, will reset ...");
                state=0;
                return 2;
              }
            } else {
                return 3;
                Serial.println("[ERROR] Exception : impossible");
            }
        }
        return 1;
    }
};


SerialReader reader;


float tanhh(float x)
{
  float x0 = exp(x);
  float x1 = 1.0 / x0;

  return ((x0 - x1) / (x0 + x1));
}


class NeuralNetwork {

private:
   JsonArray _weights;
   JsonArray _bias;



public:
    int size_input=-1;
    int size_output=-1;

   NeuralNetwork() {

   }
    void reset(){
      size_input=-1;
      size_output=-1;
      // set JsonArray to NULL
      StaticJsonDocument<1> doc;
      _weights = doc.to<JsonArray>();
      _bias = doc.to<JsonArray>();
    }

   void setWeights(JsonArray weights) {
       _weights = weights;
       size_input = weights[0][0].size();
       size_output = weights[weights.size()-1].size();
   }


   void setBias(JsonArray bias) {
       _bias = bias;
   }

   float *foward(float *x) {

       float* y =NULL;
       float* x_ = x;
       Serial.println("x_ : ");
       Serial.println(x_[0]);
       Serial.println(x_[1]);
       Serial.print("_weights.size()");Serial.println(_weights.size());
       for (int idx_layer = 0; idx_layer < _weights.size(); idx_layer++) {
           JsonArray layer_weight = _weights[idx_layer];
           Serial.print("layer_weight.size()");Serial.println(layer_weight.size());
           JsonArray layer_bias = _bias[idx_layer];
           float* y = (float *)malloc(sizeof(float) * sizeof(layer_weight));


           for (int idx_next_feat = 0; idx_next_feat < layer_weight.size(); idx_next_feat++) {
             Serial.println("------------------");
              Serial.print("idx_next_feat : ");
              Serial.println(idx_next_feat);
               float sum = 0.0;
               JsonArray feature_weight = layer_weight[idx_next_feat];
               for (int idx_weight = 0; idx_weight < feature_weight.size(); idx_weight++) {
                  Serial.println("feature_weight["+String(idx_weight)+"] : "+String(feature_weight[idx_weight].as<float>()));
                  Serial.println("x_["+String(idx_weight)+"] : "+String(x_[idx_weight]));
                  sum = sum + x_[idx_weight] * feature_weight[idx_weight].as<float>();

               }
               Serial.println("sum["+String(idx_next_feat)+"] : "+String(sum));
               y[idx_next_feat] = sum + layer_bias[idx_next_feat].as<float>();

               if(idx_layer < _weights.size() -1){
                  Serial.println("tanh !!!");
                  y[idx_next_feat] = tanhh(y[idx_next_feat]);
               }
               Serial.println("after activation["+String(idx_next_feat)+"] : "+

               String(y[idx_next_feat]));
           }

           x_=y;
       }
       /*free(x_);
       x_=NULL;*/
       return x_;
   }

};

NeuralNetwork nn;
DynamicJsonDocument doc(4000);
long loop_id = 0;
String data = "{'bias': [[0.0, 0.0]], 'weights': [[[-0.0440443754196167, -0.30511707067489624], [-0.2248501181602478, -0.06371712684631348]]]}";
DeserializationError error;

void setup() {
    Serial.begin(BAUD);
    while(!Serial);
    reader.reset();
    info("Ready to Read on baud : "+String(BAUD));
    Serial.println(" Reamining memory (setup()): ");
    Serial.println(freeMemory());

}

void loop() {
    data = reader.loop();
    if (freeMemory()<10){
      Serial.println("ALERT MEMORY !!!");
    }
    if (data != "") {
        Serial.print("looping : ");
        Serial.println(loop_id);
        Serial.print(" Reamining memory (before): ");
        Serial.print(freeMemory());
        Serial.println("");
        Serial.print("data received : ");
        Serial.println(data);
        error = deserializeJson(doc, data);// maybe add a header to the data with the exact size to not take extra memory
        nn.reset();
        if (error) {
            Serial.print(F("deserializeJson() failed: "));
            Serial.println(error.c_str());
        } else {
            nn.setWeights(doc["weights"]);
            nn.setBias(doc["bias"]);
            float* x = (float *)malloc(sizeof(float)*10);
            for(int i=0;i<nn.size_input;i++){
              x[i]=float(i);
            }

            float* y = nn.foward(x);
            Serial.print("y = ");
            for(int i=0;i <nn.size_output; i++){
              Serial.print(y[i]);
              Serial.print(" ");
            }
            Serial.println();
            Serial.print(" Reamining memory (after): ");
            Serial.println(freeMemory());
            free(y);
            y=NULL;
            free(x);
            x=NULL;
        }
        doc.clear();
        Serial.print(" Reamining memory (after): ");
        Serial.println(freeMemory());
            Serial.print(nn.size_output);Serial.print(nn.size_input);

    }
    //delay(500);
    loop_id+=1;

}
