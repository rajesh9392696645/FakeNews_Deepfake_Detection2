function showMessage(message)
{
    alert(message);
}

function confirmDelete()
{
    return confirm(
        "Are you sure you want to delete this record?"
    );
}

function validateFile(input)
{
    const file = input.files[0];

    if(!file)
    {
        return;
    }

    const size =
        file.size / (1024 * 1024);

    if(size > 100)
    {
        alert(
            "File size exceeds 100 MB"
        );

        input.value = "";
    }
}

function previewImage(event)
{
    let reader = new FileReader();

    reader.onload = function()
    {
        let output =
            document.getElementById(
                "preview"
            );

        output.src =
            reader.result;
    }

    reader.readAsDataURL(
        event.target.files[0]
    );
}