function myFunction() {
    element = document.getElementById("trigger");
    if (element.className == "trigger"){
        element.className = "show";
    }
    else {
        element.className = "trigger";
    }
}