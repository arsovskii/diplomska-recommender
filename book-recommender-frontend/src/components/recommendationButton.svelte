<script lang="ts">
	import { ratingsStore } from '$lib/stores/stores';

	let ratingsToSend: any;

	ratingsStore.subscribe((value) => {
		ratingsToSend = value;
	});

	const getRecommendations = async () => {
		try {
			const response = await fetch('http://localhost:5000/api/makeRecommendation', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ ratings: ratingsToSend })
			});

			const text = await response.text();
			console.log('Raw response:', text);

			if (response.ok) {
				const data = JSON.parse(text);

				console.log('Recommendations:', data);
			} else {
				console.error('Failed to fetch recommendations:', response.statusText);
			}
		} catch (error) {
			console.error('Error fetching recommendations:', error);
		}
	};
</script>

<div class="w-full text-center">
	<button
		class="btn btn-secondary shadow-lg mx-auto text-xl font-normal"
		on:click={getRecommendations}>Get your recommendations!</button
	>
</div>
