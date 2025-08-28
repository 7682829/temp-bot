// Import Google's Gemini chat model and embeddings for AI text generation and vector creation
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
// Import MongoDB client for database connection
import { MongoClient } from "mongodb"
// Import MongoDB Atlas vector search for storing and searching embeddings
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
// Load environment variables from .env file (API keys, connection strings)
import "dotenv/config"

// Create MongoDB client instance using connection string from environment variables
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)

// Initialize Google Gemini chat model for generating synthetic furniture data
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",  // Use Gemini 1.5 Flash model
  temperature: 0.7,               // Set creativity level (0.7 = moderately creative)
  apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment variables
})

// Initialize Google embeddings instance once (reuse to prevent memory issues)
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  modelName: "text-embedding-004",
})

// Define TypeScript interface for furniture item structure (avoiding complex Zod nesting)
interface Item {
  item_id: string;                    // Unique identifier for the item
  item_name: string;                  // Name of the furniture item
  item_description: string;           // Detailed description of the item
  brand: string;                      // Brand/manufacturer name
  manufacturer_address: {             // Nested object for manufacturer location
    street: string;                   // Street address
    city: string;                     // City name
    state: string;                    // State/province
    postal_code: string;              // ZIP/postal code
    country: string;                  // Country name
  };
  prices: {                          // Nested object for pricing information
    full_price: number;              // Regular price
    sale_price: number;              // Discounted price
  };
  categories: string[];              // Array of category tags
  user_reviews: Array<{              // Array of customer reviews
    review_date: string;             // Date of review
    rating: number;                  // Numerical rating (1-5)
    comment: string;                 // Review text comment
  }>;
  notes: string;                     // Additional notes about the item
}

// Function to create database and collection before seeding
async function setupDatabaseAndCollection(): Promise<void> {
  console.log("Setting up database and collection...")
  
  // Get reference to the inventory_database database
  const db = client.db("inventory_database")
  
  // Create the items collection if it doesn't exist
  const collections = await db.listCollections({ name: "items" }).toArray()
  
  if (collections.length === 0) {
    await db.createCollection("items")
    console.log("Created 'items' collection in 'inventory_database' database")
  } else {
    console.log("'items' collection already exists in 'inventory_database' database")
  }
}

// Function to create vector search index
async function createVectorSearchIndex(): Promise<void> {
  try {
    const db = client.db("inventory_database")
    const collection = db.collection("items")
    await collection.dropIndexes()
    const vectorSearchIdx = {
      name: "vector_index",
      type: "vectorSearch",
      definition: {
        "fields": [
          {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,
            "similarity": "cosine"
          }
        ]
      }
    }
    console.log("Creating vector search index...")
    await collection.createSearchIndex(vectorSearchIdx);

    console.log("Successfully created vector search index");
  } catch (e) {
    console.error('Failed to create vector search index:', e);
  }
}

async function generateSyntheticData(): Promise<Item[]> {
  // Create detailed prompt instructing AI to generate furniture store data
  const prompt = `You are a helpful assistant that generates furniture store item data. Generate 10 furniture store items in JSON format. Each record should include the following fields: item_id, item_name, item_description, brand, manufacturer_address (with street, city, state, postal_code, country), prices (with full_price, sale_price), categories (array), user_reviews (array with review_date, rating, comment), notes. 

  Return ONLY a valid JSON array with no additional text or markdown formatting. Ensure variety in the data and realistic values.

  Example format:
  [
    {
      "item_id": "item_001",
      "item_name": "Modern Sofa",
      "item_description": "Comfortable 3-seater sofa",
      "brand": "ComfortPlus",
      "manufacturer_address": {
        "street": "123 Factory St",
        "city": "Detroit",
        "state": "MI",
        "postal_code": "48201",
        "country": "USA"
      },
      "prices": {
        "full_price": 899,
        "sale_price": 699
      },
      "categories": ["sofa", "living room"],
      "user_reviews": [
        {
          "review_date": "2024-01-15",
          "rating": 5,
          "comment": "Very comfortable"
        }
      ],
      "notes": "Available in multiple colors"
    }
  ]`

  // Log progress to console
  console.log("Generating synthetic data...")

  // Send prompt to AI and get response
  const response = await llm.invoke(prompt)
  
  try {
    // Parse AI response as JSON
    const jsonContent = response.content as string
    // Clean the response to extract just the JSON array
    const jsonMatch = jsonContent.match(/\[[\s\S]*\]/)
    if (!jsonMatch) {
      throw new Error("No valid JSON array found in response")
    }
    return JSON.parse(jsonMatch[0]) as Item[]
  } catch (error) {
    console.error("Error parsing AI response:", error)
    console.log("Raw response:", response.content)
    // Return fallback data if parsing fails
    return generateFallbackData()
  }
}

// Fallback function to generate hardcoded data if AI parsing fails
function generateFallbackData(): Item[] {
  return [
    {
      item_id: "item_001",
      item_name: "Modern Leather Sofa",
      item_description: "A comfortable 3-seater leather sofa perfect for modern living rooms",
      brand: "ComfortPlus",
      manufacturer_address: {
        street: "123 Factory Street",
        city: "Detroit",
        state: "Michigan",
        postal_code: "48201",
        country: "USA"
      },
      prices: {
        full_price: 1299,
        sale_price: 999
      },
      categories: ["sofa", "living room", "leather"],
      user_reviews: [
        {
          review_date: "2024-01-15",
          rating: 5,
          comment: "Excellent quality and very comfortable"
        }
      ],
      notes: "Available in black, brown, and white colors"
    },
    {
      item_id: "item_002",
      item_name: "Oak Dining Table",
      item_description: "Solid oak dining table that seats 6 people comfortably",
      brand: "WoodCraft",
      manufacturer_address: {
        street: "456 Lumber Ave",
        city: "Portland",
        state: "Oregon",
        postal_code: "97201",
        country: "USA"
      },
      prices: {
        full_price: 899,
        sale_price: 699
      },
      categories: ["dining table", "dining room", "wood"],
      user_reviews: [
        {
          review_date: "2024-02-10",
          rating: 4,
          comment: "Beautiful table, well made"
        }
      ],
      notes: "Comes with matching chairs available separately"
    }
  ]
}

// Function to create a searchable text summary from furniture item data
async function createItemSummary(item: Item): Promise<string> {
  // Return Promise for async compatibility (though this function is synchronous)
  return new Promise((resolve) => {
    // Extract manufacturer country information
    const manufacturerDetails = `Made in ${item.manufacturer_address.country}`
    // Join all categories into comma-separated string
    const categories = item.categories.join(", ")
    // Convert user reviews array into readable text format
    const userReviews = item.user_reviews
      .map(
        (review) =>
          `Rated ${review.rating} on ${review.review_date}: ${review.comment}`
      )
      .join(" ")  // Join multiple reviews with spaces
    // Create basic item information string
    const basicInfo = `${item.item_name} ${item.item_description} from the brand ${item.brand}`
    // Format pricing information
    const price = `At full price it costs: ${item.prices.full_price} USD, On sale it costs: ${item.prices.sale_price} USD`
    // Get additional notes
    const notes = item.notes

    // Combine all information into comprehensive summary for vector search
    const summary = `${basicInfo}. Manufacturer: ${manufacturerDetails}. Categories: ${categories}. Reviews: ${userReviews}. Price: ${price}. Notes: ${notes}`

    // Resolve promise with complete summary
    resolve(summary)
  })
}

// Main function to populate database with AI-generated furniture data
async function seedDatabase(): Promise<void> {
  try {
    // Establish connection to MongoDB Atlas
    await client.connect()
    // Ping database to verify connection works
    await client.db("admin").command({ ping: 1 })
    // Log successful connection
    console.log("You successfully connected to MongoDB!")

    // Setup database and collection
    await setupDatabaseAndCollection()
    
    // Create vector search index
    await createVectorSearchIndex()

    // Get reference to specific database
    const db = client.db("inventory_database")
    // Get reference to items collection
    const collection = db.collection("items")

    // Clear existing data from collection (fresh start)
    await collection.deleteMany({})
    console.log("Cleared existing data from items collection")
    
    // Generate new synthetic furniture data using AI
    const syntheticData = await generateSyntheticData()
    console.log(`Generated ${syntheticData.length} items`)

    // Process items in smaller batches to reduce memory usage
    const batchSize = 3
    for (let i = 0; i < syntheticData.length; i += batchSize) {
      const batch = syntheticData.slice(i, i + batchSize)
      console.log(`Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(syntheticData.length / batchSize)}...`)
      
      // Process each item in the batch: create summary and prepare for vector storage
      const recordsWithSummaries = await Promise.all(
        batch.map(async (record) => ({
          pageContent: await createItemSummary(record),  // Create searchable summary
          metadata: {...record},                         // Preserve original item data
        }))
      )
      
      console.log(`Processing ${recordsWithSummaries.length} records with vector embeddings...`)
      
      // Store batch records with vector embeddings in MongoDB
      await MongoDBAtlasVectorSearch.fromDocuments(
        recordsWithSummaries,         // Current batch
        embeddings,                   // Reuse the single embeddings instance
        {
          collection,                 // MongoDB collection reference
          indexName: "vector_index",  // Name of vector search index
          textKey: "embedding_text",  // Field name for searchable text
          embeddingKey: "embedding",  // Field name for vector embeddings
        }
      )

      console.log(`Successfully processed & saved batch ${Math.floor(i / batchSize) + 1}`)
      
      // Force garbage collection between batches if available
      if (global.gc) {
        global.gc()
      }
      
      // Small delay between batches
      await new Promise(resolve => setTimeout(resolve, 1000))
    }
    
    // Log completion of entire seeding process
    console.log("Database seeding completed successfully!")

  } catch (error) {
    // Log any errors that occur during database seeding
    console.error("Error seeding database:", error)
  } finally {
    // Always close database connection when finished (cleanup)
    await client.close()
  }
}

// Execute the database seeding function and handle any errors
seedDatabase().catch(console.error)
