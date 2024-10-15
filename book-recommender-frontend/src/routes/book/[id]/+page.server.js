
/**
 * @param {{ params: { id: string } }} context
 */
export async function load({ params }) {

    console.log(params.id);

   
    return {
        "bookId": params.id
    };

}