$(function() {
    var effects = "animated bounce";
    var effectsEnd = "animationed oAnimationEnd mozAnimationEnd webkitAnimationEnd";

        $("a.button").hover(function(){
                $(this).addClass(effects).one(effectsEnd, function(){
                        $(this).removeClass(effects);
                });

        });

});