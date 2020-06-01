//Gioi han checkbox
// var limit = 2;
// $('input.single-checkbox').on('change', function(evt) {
//     if ($(this).siblings(':checked').length >= limit) {
//         this.checked = false;
//     }
// });

function toggle(source) {
    checkboxes = document.getElementsByName('column_data');
    for(var i=0, n=checkboxes.length;i<n;i++) {
      checkboxes[i].checked = source.checked;
    }
}

function toggle(source) {
  checkboxes = document.getElementsByName('cot');
  for(var i=0, n=checkboxes.length;i<n;i++) {
    checkboxes[i].checked = source.checked;
  }
}