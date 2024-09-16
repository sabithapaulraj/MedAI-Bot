document.addEventListener('DOMContentLoaded', function () {
    const text = "Your Personal Health and Wellness Guide";
    const typingText = document.getElementById('typing-text');
    let index = 0;

    function type() {
        if (index < text.length) {
            typingText.innerHTML += text.charAt(index);
            index++;
            setTimeout(type, 100); // Adjust speed by changing this number (in milliseconds)
        }
    }

    type();
});
