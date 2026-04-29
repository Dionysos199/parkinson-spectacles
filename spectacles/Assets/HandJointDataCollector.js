// @input float recordDuration = 20.0
// @input Component.ScriptComponent timerScript
// @input Asset.InternetModule serviceModule
var frames = [];
var isRecording = false;
var startTime = 0;
var cachedJoints = [];

var JOINT_NAMES = [
    "Wrist",
    "Thumb0", "Thumb1", "Thumb2", "Thumb3",
    "Index0", "Index1", "Index2", "Index3",
    "Middle0", "Middle1", "Middle2", "Middle3",
    "Ring0", "Ring1", "Ring2", "Ring3",
    "Pinky0", "Pinky1", "Pinky2", "Pinky3"
];

script.startRecording = function() {
    frames = [];
    cachedJoints = [];
    for (var i = 0; i < JOINT_NAMES.length; i++) {
        var joint = global.HandTracking[JOINT_NAMES[i]];
        if (joint) cachedJoints.push(joint);
    }
    print("Cached " + cachedJoints.length + " joints");
    startTime = getTime();
    isRecording = true;
    if (script.timerScript) script.timerScript.setRecordingActive(true);
    print("Recording started. Duration set to: " + script.recordDuration);
};

script.stopRecording = function() {
    isRecording = false;
    print("Recording stopped. Frames: " + frames.length);

    var actualFps = frames.length / script.recordDuration;
    print("Effective FPS: " + actualFps.toFixed(1));

    if (frames.length < 60) {
        print("WARNING: Only " + frames.length + " frames collected (need 60+). Skipping Flask.");
        if (script.timerScript) script.timerScript.setRecordingActive(false);
        return;
    }
// Normalize to roughly MediaPipe scale
var scale = 500.0;
for (var f = 0; f < frames.length; f++) {
    for (var i = 0; i < 21; i++) {
        frames[f]["x_" + i] = (parseFloat(frames[f]["x_" + i]) / scale).toFixed(4);
        frames[f]["y_" + i] = (parseFloat(frames[f]["y_" + i]) / scale).toFixed(4);
        frames[f]["z_" + i] = "0.0000";
    }
}
    var payload = JSON.stringify({
        hand: "Left",
        fps: 30,
        frames: frames
    });

    print("Sending " + payload.length + " bytes to Flask...");

    var req = RemoteServiceHttpRequest.create();
    req.url = "http://localhost:5000/analyze_keypoints";
    req.method = RemoteServiceHttpRequest.HttpRequestMethod.Post;
    req.body = payload;
    req.headers = { "Content-Type": "application/json" };

    script.serviceModule.performHttpRequest(req, function(res) {
        print("Status: " + res.statusCode);
        print("Body: " + res.body);
        if (res.statusCode === 200) {
            global.assessmentResults = JSON.parse(res.body);
            global.behaviorSystem.sendCustomTrigger("Results_Ready");
        }
        if (script.timerScript) script.timerScript.setRecordingActive(false);
    });
    print("First frame: " + JSON.stringify(frames[0]));
    print("Mid frame: " + JSON.stringify(frames[Math.floor(frames.length/2)]));
    print("Last frame: " + JSON.stringify(frames[frames.length-1]));
};

var updateEvent = script.createEvent("UpdateEvent");
updateEvent.bind(function() {
    if (!isRecording) return;

    var elapsed = getTime() - startTime;
    if (elapsed >= script.recordDuration) {
        script.stopRecording();
        return;
    }

    if (cachedJoints.length === 0) return;

    var frame = { timestamp: Math.round(elapsed * 1000) };
    for (var i = 0; i < JOINT_NAMES.length; i++) {
        var joint = global.HandTracking[JOINT_NAMES[i]];
        if (joint) {
            var pos = joint.getWorldPosition();
            frame["x_" + i] = pos.x.toFixed(4);
            frame["y_" + i] = pos.y.toFixed(4);
            frame["z_" + i] = pos.z.toFixed(4);
        }
    }
    frames.push(frame);
}); 