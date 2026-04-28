// OnDirectionDetectedLost.js
// Version: 0.1.0
// Event: On Awake
// Description: Define your own behavior to run when the palm direction is detected or lost.
//              In this example, an object is enabled and disabled and optionally flys in and out.

//@input SceneObject exampleObject
//@input vec3 offsetFromPalm = {0,10,0} {"hint":"Distance of object relative to palm"}
//@input bool flyIn = true
//@input vec3 startPosition = {0,50,0} {"showIf" : "flyIn", "hint":"The position the object starts at and flys away to"}
//@input int flyInSpeed = 10 {"showIf" : "flyIn"}
//@input SceneObject hint

var transform;

function onDirectionDetected() {
    //
    // ------ EDIT ME: Add your own behavior here
    //
    if (script.hint) {
         script.hint.enabled = false;     
    }
      
    var hand = global.handTracking.getHand();
    var position = hand.palmCenter.add(script.offsetFromPalm);   
    
    if (script.flyIn) {
       flyTo(position); 
    } else {
        transform.setWorldPosition(position);
        script.exampleObject.enabled = true;
    }
    
}

function onDirectionLost() {
    //
    // ------ EDIT ME: Add your own behavior here
    // 
    if (script.flyIn) {
       flyTo(script.startPosition); 
    } else {
       script.exampleObject.enabled = false;
    }
}

script.onDirectionDetected = onDirectionDetected;
script.onDirectionLost = onDirectionLost;

function flyTo(destination) {
    var curPos = transform.getWorldPosition();
    var newPos = vec3.lerp(curPos, destination, getDeltaTime() * script.flyInSpeed);
    transform.setWorldPosition(newPos);
}

function init() {
    transform = script.exampleObject.getTransform();
    transform.setWorldPosition(script.startPosition);
}

init();